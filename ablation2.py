from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from utils import bootstrap_mean_ci_multi, error_rate
from plot_utils import plot_panels_with_ci


# --------------------------- Config & utilities --------------------------- #

CB_PALETTE = {
    "known": "#D55E00",     # red for known-K (ours)
    "unknown": "#0072B2",   # blue for unknown-K variants
    "oracle": "#009E73",    # green for oracle
}


@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    em_iters: int
    # Sweep of (c1, c2) for unknown-K: factors for sigma and rho thresholds
    c_pairs: Tuple[Tuple[float, float], ...]
    seed: int | None = None


# --------------------------- Per-repeat run --------------------------- #

def run_one_repeat(cfg: ExperimentConfig, repeat_idx: int) -> Dict:
    """
    For a fixed (T, H), produce error for:
    - Known-K: Stage I (K known) + EM refinement (cfg.em_iters)
    - Unknown-K: Stage I (K unknown) + EM refinement, sweeping (c1, c2)

    Returns dict with scalar errors per algorithm variant.
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed + 37 * repeat_idx + 1009 * cfg.T + 17 * cfg.H)

    env_config = _build_env_config(cfg.H, cfg.K, cfg.S)
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    f_true, trajectories = env.generate_trajectories(cfg.T)
    true_arr = np.fromiter((f_true[t] for t in range(cfg.T)), dtype=int)

    # Oracle baseline
    f_oracle = OracleLikelihoodRefinement(trajectories, env)
    pred_oracle = np.fromiter((f_oracle[t] for t in range(cfg.T)), dtype=int)
    err_oracle = error_rate(pred_oracle, true_arr)

    # Known-K pipeline: Stage I and Stage I+II (EM10)
    f_hat_known = InitialSpectral(
        trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=cfg.K
    )
    pred_known_s1 = np.fromiter((f_hat_known[t] for t in range(cfg.T)), dtype=int)
    err_known_s1 = error_rate(pred_known_s1, true_arr)

    f_known_ref = LikelihoodRefinement(
        trajectories, f_hat_known, cfg.T, cfg.S, max_iter=cfg.em_iters, history=False
    )
    pred_known_em = np.fromiter((f_known_ref[t] for t in range(cfg.T)), dtype=int)
    err_known_em = error_rate(pred_known_em, true_arr)

    # Unknown-K pipeline under different (c1, c2)
    unknown_errors: Dict[str, float] = {}
    for (c1, c2) in cfg.c_pairs:
        # Normalize float keys to scientific notation (e.g., 1e-04), for consistent legend labels
        c1s = format(c1, '.0e')
        c2s = format(c2, '.0e')

        # Stage I (unknown K)
        f_hat_unknown = InitialSpectral(
            trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=None, c1=c1, c2=c2
        )
        pred_unknown_s1 = np.fromiter((f_hat_unknown[t] for t in range(cfg.T)), dtype=int)
        unknown_errors[f"unknown_c1{c1s}_c2{c2s}_stage1"] = error_rate(pred_unknown_s1, true_arr)

        # Stage I+II (EM10) for unknown K
        f_unknown_ref = LikelihoodRefinement(
            trajectories, f_hat_unknown, cfg.T, cfg.S, max_iter=cfg.em_iters, history=False
        )
        pred_unknown_em = np.fromiter((f_unknown_ref[t] for t in range(cfg.T)), dtype=int)
        unknown_errors[f"unknown_c1{c1s}_c2{c2s}_em{cfg.em_iters}"] = error_rate(pred_unknown_em, true_arr)

    out = {
        "T": cfg.T,
        "H": cfg.H,
        "repeat": repeat_idx,
        "oracle": err_oracle,
        "known_stage1": err_known_s1,
        f"known_em{cfg.em_iters}": err_known_em,
    }
    out.update(unknown_errors)
    return out


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 3,
    S: int = 10,
    delta: float = 0.05,
    em_iters: int = 10,
    c_pairs: Sequence[Tuple[float, float]] = ((1e-4, 1e-1), (5e-5, 5e-2), (2e-4, 2e-1)),
    n_repeat: int = 20,
    n_jobs: int = -1,
    base_seed: int | None = 2025,
) -> pd.DataFrame:
    """Run all (T,H) × repeats and return a tidy DataFrame of per-repeat errors."""
    jobs: List[ExperimentConfig] = []
    for T in T_list:
        for H in H_list:
            for r in range(n_repeat):
                jobs.append(
                    ExperimentConfig(
                        T=T, H=H, K=K, S=S, delta=delta,
                        em_iters=em_iters,
                        c_pairs=tuple(c_pairs),
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
    Bootstrap mean & CIs for each (T,H,algorithm). Returns tidy DataFrame:
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

def _style_for(algo: str) -> Tuple[str, str]:
    # Fallback global style (unused by plot_panels which builds a local style with colors per (c1,c2))
    if algo == "oracle":
        return CB_PALETTE["oracle"], ":"
    if algo == "known_stage1":
        return CB_PALETTE["known"], "--"
    if algo.startswith("known_em"):
        return CB_PALETTE["known"], "-"
    if algo.startswith("unknown_c1"):
        return CB_PALETTE["unknown"], (0, (2, 2)) if algo.endswith("stage1") else "-"
    return "#1F77B4", (0, (2, 2))


def plot_panels(summary: pd.DataFrame, savepath: str = "results/ablation2_results.pdf") -> None:
    # Series and labels
    all_algos = [c for c in summary["algo"].unique()]
    unknown_s1 = sorted([a for a in all_algos if a.startswith("unknown_c1") and a.endswith("stage1")])
    # Find em label suffix dynamically (e.g., em10)
    unknown_em = sorted([a for a in all_algos if a.startswith("unknown_c1") and ("_em" in a) and not a.endswith("stage1")])

    # Derive EM suffix from known key (assumes consistent em_iters)
    known_em_keys = sorted([a for a in all_algos if a.startswith("known_em")])
    em_suffix = known_em_keys[0].split("known_")[1] if known_em_keys else "em10"

    # Interleave unknown pairs: for legend vertical alignment, we will build legend_order.
    series_order = ["oracle", "known_stage1", f"known_{em_suffix}"] + unknown_s1 + unknown_em

    label_map = {
        "oracle": "Oracle",
        "known_stage1": "Known K (Stage I)",
        f"known_{em_suffix}": f"Known K ({em_suffix.upper()})",
    }
    # Build pair labels for unknown variants
    def pretty_pair(a: str) -> str:
        # a like: unknown_c11e-06_c21e-03_stage1 or _em10; show in 1e-? format
        base = a.split("_stage1")[0].split("_em")[0]
        _, c1s, c2s = base.split("_")  # ["unknown", "c11e-04", "c21e-03"]
        c1v = c1s.replace("c1", "")
        c2v = c2s.replace("c2", "")
        return f"c1={c1v}, c2={c2v}"

    for a in unknown_s1:
        label_map[a] = f"Unknown ({pretty_pair(a)}) Stage I"
    for a in unknown_em:
        label_map[a] = f"Unknown ({pretty_pair(a)}) {a.split('_')[-1].upper()}"

    # Legend alignment: two rows, columns grouped per variant
    # Build legend labels in column-major order: [oracle, spacer], [known S1, known EM], then one col per unknown pair
    legend_cols: List[List[str]] = []
    legend_cols.append(["Oracle", "<spacer>"])
    legend_cols.append(["Known K (Stage I)", f"Known K ({em_suffix.upper()})"])
    # For each unknown pair, find its S1 and EM labels
    for s1 in unknown_s1:
        base = s1.split("_stage1")[0]
        em_key = [e for e in unknown_em if e.startswith(base + "_em")]
        em_label = label_map[em_key[0]] if em_key else f"Unknown ({pretty_pair(s1)}) {em_suffix.upper()}"
        legend_cols.append([label_map[s1], em_label])
    # Flatten to the expected legend_order list
    legend_order: List[str] = [lbl for col in legend_cols for lbl in col]

    # Build distinct colors per (c1,c2) base
    unknown_bases = [s1.split("_stage1")[0] for s1 in unknown_s1]
    # Palette of visually distinct blues/purples (avoid green/red used by oracle/known)
    unknown_palette = [
        "#0072B2",  # blue
        "#56B4E9",  # light blue
        "#2F65A2",  # dark blue
        "#3F87C5",  # medium blue
        "#6A51A3",  # purple
        "#9E9AC8",  # light purple
        "#264E86",  # deep blue
        "#6BAED6",  # sky blue
        "#9ECAE1",  # pale blue
        "#6A3D9A",  # purple alt
    ]
    color_map = {base: unknown_palette[i % len(unknown_palette)] for i, base in enumerate(unknown_bases)}

    # Local style function capturing color_map
    def style_for_local(algo: str) -> Tuple[str, str]:
        if algo == "oracle":
            return CB_PALETTE["oracle"], ":"
        if algo == "known_stage1":
            return CB_PALETTE["known"], "--"
        if algo.startswith("known_em"):
            return CB_PALETTE["known"], "-"
        if algo.startswith("unknown_c1"):
            base = algo.split("_stage1")[0].split("_em")[0]
            color = color_map.get(base, CB_PALETTE["unknown"])  # fallback if needed
            if algo.endswith("stage1"):
                # dashed per pair
                return color, (0, (4, 2))
            return color, "-"
        return "#1F77B4", (0, (2, 2))

    plot_panels_with_ci(
        summary,
        series_order=series_order,
        label_map=label_map,
        style_for=style_for_local,
        savepath=savepath,
        facet_col="T",
        x_col="H",
        series_col="algo",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=True,
        inset_xlim=(10, 100),
        inset_xticks=list(range(10, 101, 10)),
        extra_xticks=[300],
        y_label="Clustering error",
        legend_order=legend_order,
        legend_ncol=len(legend_cols),
    )


# --------------------------- CLI entry --------------------------- #

def main():
    T_list = [100, 200, 300, 400, 500, 600]
    H_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Choose a small set of (c1,c2) variations around defaults in Clustering.py
    c_pairs = (
        (1e-2, 1e-3),   # baseline as in current constants
        (1e-3, 1e-2),   # more permissive thresholds => works for T=100
        (1e-2, 1e-1),   # stricter thresholds
        # (1e-3, 1.0),   # stricter thresholds
        # (1e-4, 1e-3),   # baseline as in current constants
        # (1e-4, 1e-2),   # more permissive thresholds
        # (1e-4, 1e-1),   # stricter thresholds
        # (1e-4, 1.0),   # stricter thresholds
    )

    df = run_grid(
        T_list=T_list,
        H_list=H_list,
        K=3,
        S=10,
        delta=0.05,
        em_iters=10,
        c_pairs=c_pairs,
        n_repeat=30,
        n_jobs=-1,
        base_seed=2025,
    )
    df.to_csv("results/ablation2_results_unknownK_raw.csv", index=False)

    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/ablation2_results_unknownK_summary.csv", index=False)

    plot_panels(summary, savepath="results/ablation2_results.png")


if __name__ == "__main__":
    main()
