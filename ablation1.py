from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from plot_utils import plot_panels_with_ci

from Synthetic import MixtureMarkovChains
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from utils import bootstrap_mean_ci_multi, error_rate


# --------------------------- Config & utilities --------------------------- #

CB_PALETTE = {
    # Match the visual language used in main_synthetic.py
    # Ours (EM variants) in reds, Stage I a lighter red/orange, Oracle green
    "oracle": "#009E73",   # Okabe–Ito green (color-blind friendly)
    "stage1": "#E5864A",   # Light-but-visible red/orange (aligns with ours_stage1)
    "em1":   "#D55E00",    # Vermilion red (most visible)
    "em10":  "#C24700",    # Medium red
    "em20":  "#A63A00",    # Darker red for additional EM variant
}


@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    em_iters_list: Tuple[int, ...]
    seed: int | None = None


def _build_env_config(H: int, K: int, S: int) -> Dict:
    """Match the special K=2 construction used in main_synthetic.py; otherwise randomize."""
    if K == 2:
        if S % 2:
            S += 1
        half = S // 2
        # half = 8
        idx = np.arange(S)
        col_mask = (idx[None, :] >= half).astype(float)
        right = np.tile(col_mask, (S, 1))
        left = 1.0 - right

        # Use the same weighting as main_synthetic.py (rows normalized below)
        P1 = right * 2 + left * 1
        P2 = right * 1 + left * 2
        P1 /= P1.sum(axis=1, keepdims=True)
        P2 /= P2.sum(axis=1, keepdims=True)
        return {"H": H, "K": K, "S": S, "Ps": [P1, P2],
                "mus": [np.ones(S) / S, np.ones(S) / S]}
    return {"H": H, "K": K, "S": S}


# --------------------------- Core per-run work --------------------------- #

def run_one_repeat(cfg: ExperimentConfig, repeat_idx: int) -> Dict:
    """
    Run one repeat for a fixed (T, H), returning per-algorithm error rates.
    Algorithms: Oracle, Stage I, Stage I+II with EM iters in em_iters_list.
    """
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
    pred_s1 = np.fromiter((f_hat_1[t] for t in range(cfg.T)), dtype=int)
    err_s1 = error_rate(pred_s1, true_arr)

    # Stage I + II with various EM iterations
    em_errs: Dict[int, float] = {}
    for iters in cfg.em_iters_list:
        f_ref = LikelihoodRefinement(trajectories, f_hat_1, cfg.T, cfg.S, max_iter=iters)
        pred_ref = np.fromiter((f_ref[t] for t in range(cfg.T)), dtype=int)
        em_errs[iters] = error_rate(pred_ref, true_arr)

    out = {
        "T": cfg.T,
        "H": cfg.H,
        "repeat": repeat_idx,
        "oracle": err_oracle,
        "stage1": err_s1,
    }
    for iters, e in em_errs.items():
        out[f"em{iters}"] = e
    return out


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 2,
    S: int = 10,
    delta: float = 0.05,
    em_iters_list: Sequence[int] = tuple(range(1, 11)),
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
                        em_iters_list=tuple(em_iters_list),
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

def plot_panels(summary: pd.DataFrame, savepath: str = "results/ablation1_results.pdf",
                inset: bool = True,
                inset_xlim: Tuple[int, int] | None = None,
                inset_xticks: Sequence[int] | None = None) -> None:
    # Prepare a smooth color spectrum for EM1..EM10 using a Reds colormap
    em_series = [f"em{i}" for i in range(1, 11)]
    reds_cmap = plt.get_cmap("Reds")
    # Avoid very light and very dark extremes for better visibility
    em_color_values = np.linspace(0.35, 0.85, len(em_series))
    EM_COLORS = {name: reds_cmap(val) for name, val in zip(em_series, em_color_values)}

    def style_for(algo: str) -> Tuple[str, str]:
        if algo == "oracle":
            # Use dotted style for Oracle as in main_synthetic
            return CB_PALETTE["oracle"], ":"
        if algo == "stage1":
            return CB_PALETTE["stage1"], "--"
        # Any EM variant em1..em10 gets a solid line with gradient color
        if algo.startswith("em"):
            return EM_COLORS.get(algo, "#C24700"), "-"
        # Fallback to a subtle dashed blue
        return "#1F77B4", (0, (2, 2))

    label_map = {
        "oracle": "Oracle",
        "stage1": "Ours (Stage I)",
        **{f"em{i}": f"Ours (Stage I+II, {i} EM)" for i in range(1, 11)},
    }

    algo_order = ["oracle", "stage1"] + [f"em{i}" for i in range(1, 11)]

    plot_panels_with_ci(
        summary,
        series_order=algo_order,
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
        extra_xticks=[10, 20, 30, 40, 50],
        # Larger fonts
        title_fontsize=18,
        label_fontsize=16,
        tick_fontsize=14,
        legend_fontsize=16,
    )


# --------------------------- CLI entry --------------------------- #

def main():
    T_list = [100, 200, 300, 400, 500, 600]
    H_list = [10, 20, 30, 40, 50]

    df = run_grid(
        T_list=T_list, H_list=H_list,
        K=3, S=10, delta=0.05,
        em_iters_list=tuple(range(1, 11)),
        n_repeat=30,
        n_jobs=-1,
        base_seed=2025,
    )
    df.to_csv("results/ablation1_results_raw.csv", index=False)

    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/ablation1_results_summary.csv", index=False)

    # Plot panels with an inset focusing on H=10..100 (ticks at 10,20,...,100)
    inset_ticks = list(range(10, 21, 10))
    plot_panels(
        summary,
        savepath="results/ablation1_results.pdf",
        inset=True,
        inset_xlim=(10, 20),
        inset_xticks=inset_ticks,
    )


if __name__ == "__main__":
    main()


