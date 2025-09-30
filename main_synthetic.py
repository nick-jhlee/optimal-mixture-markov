from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from mdpmix_three_stage import MixtureMarkovChainLearner
from utils import bootstrap_mean_ci_multi, error_rate
from plot_utils import plot_panels_with_ci


# --------------------------- Config & utilities --------------------------- #

CB_PALETTE = {
    # --- Ours (reds): strongest = EM1 ---
    "ours_em1":    "#D55E00",  # Vermilion red (most visible)
    "ours_em10":   "#C24700",  # Medium red
    "ours_stage1": "#E5864A",  # Light-but-still-visible red/orange

    # --- Oracle ---
    "oracle":      "#009E73",  # Okabe–Ito green (color-blind friendly)

    # Kausik (blues, LIGHTENED)
    "kausik_p10":  "#2F65A2",  # lighter navy (was #1F4E79)
    "kausik_p30":  "#3F87C5",  # medium-light blue (was #0072B2)
    "kausik_p50":  "#85C3E9",  # softer light blue (was #56B4E9)
}

STYLES = {
    "ours_em1":    dict(lw=2.8, linestyle="-",  marker="o", zorder=4),
    "ours_em10":   dict(lw=2.4, linestyle="-",  marker="s", zorder=3),
    "ours_stage1": dict(lw=2.2, linestyle=":",  marker=".", zorder=2),

    "oracle":      dict(lw=2.6, linestyle="--", marker="*", zorder=3),

    "kausik_p10":  dict(lw=2.0, linestyle="--", marker="^", zorder=1),
    "kausik_p30":  dict(lw=2.0, linestyle="--", marker="D", zorder=1),
    "kausik_p50":  dict(lw=2.0, linestyle="--", marker="v", zorder=1),
}

CI_ALPHA = 0.15

@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    percentiles: Tuple[int, ...]
    em_iters: int = 1
    mdpmix_em_iters: int = 50
    seed: int | None = None



# --------------------------- Core per-run work --------------------------- #

def run_one_repeat(cfg: ExperimentConfig, repeat_idx: int) -> Dict:
    """
    Run one repeat for a fixed (T, H), returning per-algorithm error rates.
    Returns a dict with scalar errors so we can DataFrame-ify easily.
    """
    # Per-repeat deterministic seed if base provided
    if cfg.seed is not None:
        np.random.seed(cfg.seed + 37 * repeat_idx + 1009 * cfg.T + 17 * cfg.H)

    env_config = _build_env_config(cfg.H, cfg.K, cfg.S)
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    f_true, trajectories = env.generate_trajectories(cfg.T)
    true_arr = np.fromiter((f_true[t] for t in range(cfg.T)), dtype=int)

    # Oracle
    f_oracle = OracleLikelihoodRefinement(trajectories, env)
    pred_oracle = np.fromiter((f_oracle[t] for t in range(cfg.T)), dtype=int)
    err_oracle = error_rate(pred_oracle, true_arr)

    # Stage I
    f_hat_1 = InitialSpectral(trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=cfg.K)
    pred1 = np.fromiter((f_hat_1[t] for t in range(cfg.T)), dtype=int)
    err_s1 = error_rate(pred1, true_arr)

    # Ours (Stage I+II), compute 1 EM and 10 EM from a single history run
    f_hat_hist, _ = LikelihoodRefinement(trajectories, f_hat_1, cfg.T, cfg.S, max_iter=10, history=True)
    pred_em1 = np.fromiter((f_hat_hist[0][t] for t in range(cfg.T)), dtype=int)
    err_em1 = error_rate(pred_em1, true_arr)
    pred_em10 = np.fromiter((f_hat_hist[-1][t] for t in range(cfg.T)), dtype=int)
    err_em10 = error_rate(pred_em10, true_arr)

    # Kausik mdpmix at multiple percentiles
    mdpmix_errs = {}
    for p in cfg.percentiles:
        learner = MixtureMarkovChainLearner(K=cfg.K, verbose=False)
        learner.fit(
            trajectories,
            percentile=p,
            max_em_iterations=cfg.mdpmix_em_iters,
            plot_histogram=False,
        )
        pred = np.array([int(learner.cluster_labels[t]) for t in range(cfg.T)], dtype=int)
        mdpmix_errs[p] = error_rate(pred, true_arr)

    out = {
        "T": cfg.T,
        "H": cfg.H,
        "repeat": repeat_idx,
        "oracle": err_oracle,
        "stage1": err_s1,
        "ours_em1": err_em1,
        "ours_em10": err_em10,
    }
    for p, e in mdpmix_errs.items():
        out[f"mdpmix_p{p}"] = e
    return out


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 2,
    S: int = 10,
    delta: float = 0.05,
    percentiles: Sequence[int] = (10, 30, 50),
    n_repeat: int = 20,
    n_jobs: int = -1,
    base_seed: int | None = 1234,
) -> pd.DataFrame:
    """Run all (T,H) × repeats and return a tidy DataFrame of per-repeat errors."""
    jobs: List[ExperimentConfig] = []
    for T in T_list:
        for H in H_list:
            for r in range(n_repeat):
                jobs.append(
                    ExperimentConfig(
                        T=T, H=H, K=K, S=S, delta=delta,
                        percentiles=tuple(percentiles),
                        em_iters=1,
                        mdpmix_em_iters=50,
                        seed=base_seed,
                    )
                )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_repeat)(cfg, r) for r, cfg in enumerate(jobs)
    )
    df = pd.DataFrame(results).sort_values(["T", "H", "repeat"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Bootstrap mean & CIs for each (T,H,algorithm). Returns a tidy DataFrame:
    columns = [T, H, algo, mean, lo, hi]
    """
    algos = [c for c in df.columns if c not in ("T", "H", "repeat")]
    rows = []
    for (T, H), g in df.groupby(["T", "H"], sort=True):
        for algo in algos:
            vals = g[algo].to_numpy()
            means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
            rows.append(
                {"T": T, "H": H, "algo": algo,
                 "mean": float(means[0]),
                 "lo": float(lo[0]),
                 "hi": float(hi[0])}
            )
    out = pd.DataFrame(rows).sort_values(["T", "H", "algo"]).reset_index(drop=True)
    return out


# --------------------------- Plotting --------------------------- #

def plot_panels(summary: pd.DataFrame, percentiles: Sequence[int], savepath: str = "results/main_synthetic_results.pdf",
                inset: bool = True,
                inset_xlim: Tuple[int, int] | None = None,
                inset_xticks: Sequence[int] | None = None) -> None:
    def style_for(algo: str) -> Tuple[str, str]:
        if algo == "oracle":
            return CB_PALETTE["oracle"], ":"
        if algo == "stage1":
            return CB_PALETTE["ours_stage1"], "--"
        if algo == "ours_em1":
            return CB_PALETTE["ours_em1"], "-"
        if algo == "ours_em10":
            return CB_PALETTE["ours_em10"], "-"
        # Distinguish Kausik variants per-percentile with varying dash patterns
        # Example patterns indexed by percentile lexicographic order
        dash_patterns = [
            (0, (2, 2)),   # short dots
            (0, (4, 2)),   # slightly longer dashes
            (0, (6, 2)),   # long dashes
            (0, (4, 1, 1, 1)),  # dash-dot
            (0, (3, 1, 1, 1, 1, 1)),  # complex pattern for extra variants
        ]
        try:
            if algo.startswith("mdpmix_p"):
                p = int(algo.split("_p")[1])
                idx = sorted(percentiles).index(p) % len(dash_patterns)
                color = CB_PALETTE.get(f"kausik_p{p}", "#1F77B4")
                return color, dash_patterns[idx]
        except Exception:
            pass
        return "#1F77B4", (0, (2, 2))

    label_map = {
        "oracle": "Oracle",
        "stage1": "Ours (Stage I)",
        "ours_em1": "Ours (Stage I+II, 1 EM)",
        "ours_em10": "Ours (Stage I+II, 10 EM)",
        **{f"mdpmix_p{p}": f"Kausik (p{p})" for p in percentiles},
    }

    # Ensure Kausik variants are visually distinct: alternate dash patterns
    series_order = ["oracle", "stage1", "ours_em1", "ours_em10"] + [f"mdpmix_p{p}" for p in percentiles]

    # Previous two-row legend layout using spacers (disabled, kept for reference):
    # legend_order = [
    #     "Oracle",                    # 1 (col1,row1)
    #     "<spacer>",                 # 2 (col1,row2)
    #     "Stage I",                  # 3 (col2,row1)
    #     "<spacer>",                 # 4 (col2,row2)
    #     "Ours (Stage I+II, 1 EM)",  # 5 (col3,row1)
    #     "Ours (Stage I+II, 10 EM)", # 6 (col3,row2)
    #     "Kausik (p10)",             # 7 (col4,row1)
    #     "Kausik (p30)",             # 8 (col4,row2)
    #     "Kausik (p50)",             # 9 (col5,row1)
    #     "<spacer>",                 # 10 (col5,row2)
    # ]
    # Single-row legend: use as many columns as series
    legend_order = None
    legend_ncol = len(series_order)

    plot_panels_with_ci(
        summary,
        series_order=series_order,
        label_map=label_map,
        style_for=style_for,
        savepath=savepath,
        facet_col="T",
        x_col="H",
        series_col="algo",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=inset,
        inset_xlim=inset_xlim,
        inset_xticks=inset_xticks,
        legend_order=legend_order,
        legend_ncol=legend_ncol,
        extra_xticks=[300, 500],
        # Pull legend a bit closer and reduce top padding
        legend_bbox=(0.5, 1.06),
        top_adjust=0.84,
        # Larger fonts
        title_fontsize=18,
        label_fontsize=16,
        tick_fontsize=14,
        legend_fontsize=16,
    )


# --------------------------- CLI entry --------------------------- #

def main():
    # T_list = [100, 200, 300, 400, 500, 600]
    T_list = [100, 200, 300, 400, 500, 600]
    # H_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    H_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    percentiles = (10, 30, 50)

    df = run_grid(
        T_list=T_list, H_list=H_list,
        K=3, S=10, delta=0.05,
        percentiles=percentiles,
        n_repeat=30,   # adjust
        n_jobs=-1,
        base_seed=2025,
    )
    # Save raw results
    df.to_csv("results/main_synthetic_results_raw.csv", index=False)

    # Bootstrap & save summary
    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/main_synthetic_results_summary.csv", index=False)

    # Plot panels
    # Example: focus inset on H=10..100 with ticks at 10,20,...,100
    inset_ticks = list(range(10, 101, 10))
    plot_panels(
        summary,
        percentiles,
        savepath="results/main_synthetic_results.pdf",
        inset=True,
        inset_xlim=(10, 100),
        inset_xticks=inset_ticks,
    )


if __name__ == "__main__":
    main()
