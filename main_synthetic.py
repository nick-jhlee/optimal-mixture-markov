from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from mcmix.kausik_original import run_kausik_original
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

    # Kausik / mdpmix (blues)
    "kausik_th0":  "#2F65A2",
    "kausik_th1":  "#3F87C5",
    "kausik_th2":  "#85C3E9",
}

STYLES = {
    "ours_em1":    dict(lw=2.8, linestyle="-",  marker="o", zorder=4),
    "ours_em10":   dict(lw=2.4, linestyle="-",  marker="s", zorder=3),
    "ours_stage1": dict(lw=2.2, linestyle=":",  marker=".", zorder=2),

    "oracle":      dict(lw=2.6, linestyle="--", marker="*", zorder=3),

    "kausik_th0":  dict(lw=2.0, linestyle="--", marker="^", zorder=1),
    "kausik_th1":  dict(lw=2.0, linestyle="--", marker="D", zorder=1),
    "kausik_th2":  dict(lw=2.0, linestyle="--", marker="v", zorder=1),
}

CI_ALPHA = 0.15

def _fmt_th(th: float) -> str:
    return f"{float(th):.0e}"


def _mdpmix_original_error(
    trajectories: List[List[int]],
    true_arr: np.ndarray,
    *,
    K: int,
    S: int,
    H: int,
    mdpmix_thresh: float,
    em_iters: int,
    mdpmix_em_laplace: float = 0.0,
) -> Tuple[float, float]:
    out = run_kausik_original(
        trajectories,
        true_arr,
        K=K,
        S=S,
        H=H,
        mdpmix_thresh=mdpmix_thresh,
        mdpmix_em_iters=em_iters,
        mdpmix_em_laplace=mdpmix_em_laplace,
        repo_root=Path(__file__).resolve().parent,
    )
    return float(out["mdpmix_original_clust"]), float(out["mdpmix_original_em"])

@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    mdpmix_threshes: Tuple[float, ...]
    em_iters: int = 1
    mdpmix_em_iters: int = 10
    mdpmix_em_laplace: float = 0.0
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

    # Kausik mdpmix at multiple thresholds using original mcmix implementation
    mdpmix_errs = {}
    for th in cfg.mdpmix_threshes:
        mdpmix_errs[th] = _mdpmix_original_error(
            trajectories,
            true_arr,
            K=cfg.K,
            S=cfg.S,
            H=cfg.H,
            mdpmix_thresh=float(th),
            em_iters=cfg.mdpmix_em_iters,
            mdpmix_em_laplace=cfg.mdpmix_em_laplace,
        )

    out = {
        "T": cfg.T,
        "H": cfg.H,
        "repeat": repeat_idx,
        "oracle": err_oracle,
        "stage1": err_s1,
        "ours_em1": err_em1,
        "ours_em10": err_em10,
    }
    for th, (e_clust, e_em) in mdpmix_errs.items():
        th_tag = _fmt_th(th)
        out[f"mdpmix_original_clust_th{th_tag}"] = e_clust
        out[f"mdpmix_original_em_th{th_tag}"] = e_em
    return out


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 2,
    S: int = 10,
    delta: float = 0.05,
    mdpmix_threshes: Sequence[float] = (1e-5, 5e-5, 1e-4),
    mdpmix_em_laplace: float = 0.0,
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
                        mdpmix_threshes=tuple(mdpmix_threshes),
                        em_iters=1,
                        mdpmix_em_iters=50,
                        mdpmix_em_laplace=mdpmix_em_laplace,
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

def plot_panels(summary: pd.DataFrame, mdpmix_threshes: Sequence[float], savepath: str = "results/main_synthetic_results.pdf",
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
        # Kausik: same color per threshold; dotted=clust, solid=EM.
        th_tokens = [_fmt_th(th) for th in sorted(mdpmix_threshes)]
        try:
            if algo.startswith("mdpmix_original_clust_th"):
                th = algo.split("_th", 1)[1]
                idx = th_tokens.index(th) % 3
                color = CB_PALETTE.get(f"kausik_th{idx}", "#1F77B4")
                return color, ":"
            if algo.startswith("mdpmix_original_em_th"):
                th = algo.split("_th", 1)[1]
                idx = th_tokens.index(th) % 3
                color = CB_PALETTE.get(f"kausik_th{idx}", "#1F77B4")
                return color, "-"
        except Exception:
            pass
        return "#1F77B4", (0, (2, 2))

    label_map = {
        "oracle": "Oracle",
        "stage1": "Ours (Stage I)",
        "ours_em1": "Ours (Stage I+II, 1EM)",
        "ours_em10": "Ours (Stage I+II, 10EM)",
        **{f"mdpmix_original_clust_th{_fmt_th(th)}": f"Kausik (th{_fmt_th(th)})" for th in mdpmix_threshes},
        **{f"mdpmix_original_em_th{_fmt_th(th)}": f"Kausik (th{_fmt_th(th)}+EM)" for th in mdpmix_threshes},
    }

    series_order = ["oracle", "stage1", "ours_em1", "ours_em10"]
    for th in mdpmix_threshes:
        th_tag = _fmt_th(th)
        series_order.extend([f"mdpmix_original_clust_th{th_tag}", f"mdpmix_original_em_th{th_tag}"])

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
    # Two-row legend layout (10 total series -> 5 columns x 2 rows)
    legend_order = None
    legend_ncol = 5

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
        main_xticks=[10, 50, 90, 200, 300, 400, 500],
        # Pull legend a bit closer and reduce top padding
        legend_bbox=(0.5, 1.08),
        top_adjust=0.82,
        # Larger fonts
        title_fontsize=22,
        label_fontsize=20,
        tick_fontsize=18,
        legend_fontsize=20,
    )


# --------------------------- CLI entry --------------------------- #

def main():
    # T_list = [100, 200, 300, 400, 500, 600]
    T_list = [100, 200, 300, 400, 500, 600]
    # H_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    H_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    mdpmix_threshes = (1e-5, 1e-4, 1e-3)
    mdpmix_em_laplace = 0.1

    df = run_grid(
        T_list=T_list, H_list=H_list,
        K=3, S=10, delta=0.05,
        mdpmix_threshes=mdpmix_threshes,
        mdpmix_em_laplace=mdpmix_em_laplace,
        n_repeat=30,   # adjust
        n_jobs=-1,
        base_seed=2025,
    )
    # Save raw results
    df.to_csv("results/main_synthetic_results_raw.csv", index=False)

    # Bootstrap & save summary
    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/main_synthetic_results_summary.csv", index=False)

    # Plot panels - save both PDF and PNG
    # Inset plotting disabled for cleaner panels.
    # Example (disabled): focus inset on H=10..100 with ticks at 10,20,...,100
    # inset_ticks = list(range(10, 101, 10))
    plot_panels(
        summary,
        mdpmix_threshes,
        savepath="results/main_synthetic_results.pdf",
        inset=False,
        # inset=True,
        # inset_xlim=(10, 100),
        # inset_xticks=inset_ticks,
    )
    plot_panels(
        summary,
        mdpmix_threshes,
        savepath="results/main_synthetic_results.png",
        inset=False,
        # inset=True,
        # inset_xlim=(10, 100),
        # inset_xticks=inset_ticks,
    )


if __name__ == "__main__":
    main()
