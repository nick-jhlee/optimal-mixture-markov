from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from plot_utils import plot_panels_with_ci

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement, clustered_transition_matrix
from utils import bootstrap_mean_ci_multi, error_rate, transition_matrix_error, log_likelihood


# --------------------------- Config & utilities --------------------------- #

CB_PALETTE = {
    "clustering": "#0072B2",  # blue
    "transition": "#D55E00",  # reddish
    "loglik": "#999999",      # gray
    "oracle": "#009E73",      # green
}


@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    max_iter: int
    seed: int | None = None



# --------------------------- Core per-run work --------------------------- #

def run_one_repeat(cfg: ExperimentConfig, repeat_idx: int) -> List[Dict]:
    """
    For a fixed (T, H), return per-iteration metrics for Stage I+II:
    - clustering: error rate vs iteration
    - transition: transition matrix error vs iteration

    Returns a list of rows with keys: T, H, repeat, iter, clustering, transition
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed + 37 * repeat_idx + 1009 * cfg.T + 17 * cfg.H)

    env_config = _build_env_config(cfg.H, cfg.K, cfg.S)
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    print(f"D_pi: {env.D_pi}")
    print(f"Delta_W: {env.Delta_W}")
    f_true, trajectories = env.generate_trajectories(cfg.T)
    P_true = np.array(env.Ps)

    # Oracle clustering error (constant across iterations)
    f_oracle = OracleLikelihoodRefinement(trajectories, env)
    oracle_err = error_rate(f_oracle, f_true)

    # Stage I baseline (iteration 0)
    f_hat_1 = InitialSpectral(trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=cfg.K)
    pred1 = np.array([int(f_hat_1[t]) for t in range(cfg.T)], dtype=int)
    pred1_dict = {t: int(pred1[t]) for t in range(cfg.T)}
    clustering_errs: List[float] = [error_rate(pred1_dict, f_true)]
    P_est_1 = np.array([
        clustered_transition_matrix(trajectories, [t for t in range(cfg.T) if f_hat_1[t] == k], cfg.S)
        for k in range(cfg.K)
    ])
    transition_errs: List[float] = [transition_matrix_error(P_est_1, P_true)]

    # Oracle total likelihood using true transition matrices
    oracle_total_likelihood = 0.0
    for trajectory in trajectories:
        likelihoods_true = [log_likelihood(trajectory, P) for P in P_true]
        oracle_total_likelihood += float(np.max(likelihoods_true) / (cfg.T * cfg.H))

    # Baseline total likelihood at iteration 0 using P_est_1
    total_likelihood_0 = 0.0
    for trajectory in trajectories:
        likelihoods_t = [log_likelihood(trajectory, P) for P in P_est_1]
        total_likelihood_0 += float(np.max(likelihoods_t) / (cfg.T * cfg.H))
    logliks: List[float] = [total_likelihood_0]

    # Stage II refinement history up to max_iter
    f_hat_2s, iter_logliks = LikelihoodRefinement(trajectories, f_hat_1, cfg.T, cfg.S, max_iter=cfg.max_iter, history=True)
    for f_hat_2, ll in zip(f_hat_2s, iter_logliks.tolist()):
        pred = np.array([int(f_hat_2[t]) for t in range(cfg.T)], dtype=int)
        pred_dict = {t: int(pred[t]) for t in range(cfg.T)}
        clustering_errs.append(error_rate(pred_dict, f_true))

        P_est_2 = np.array([
            clustered_transition_matrix(trajectories, [t for t in range(cfg.T) if f_hat_2[t] == k], cfg.S)
            for k in range(cfg.K)
        ])
        transition_errs.append(transition_matrix_error(P_est_2, P_true))

        # Normalize total likelihood per user's scheme
        logliks.append(float(ll))

    rows: List[Dict] = []
    for i in range(len(clustering_errs)):
        rows.append({
            "T": cfg.T,
            "H": cfg.H,
            "repeat": repeat_idx,
            "iter": i,  # 0 = Stage I, then 1..max_iter
            "clustering": clustering_errs[i],
            "transition": transition_errs[i],
            "loglik": logliks[i],
            "oracle_loglik": oracle_total_likelihood,
            "oracle": oracle_err,
        })
    return rows


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 2,
    S: int = 10,
    delta: float = 0.05,
    max_iter: int = 20,
    n_repeat: int = 1,
    n_jobs: int = -1,
    base_seed: int | None = 1234,
) -> pd.DataFrame:
    """Run all (T,H) × repeats and return a tidy per-iteration DataFrame."""
    jobs: List[ExperimentConfig] = []
    for T in T_list:
        for H in H_list:
            for r in range(n_repeat):
                jobs.append(
                    ExperimentConfig(
                        T=T, H=H, K=K, S=S, delta=delta,
                        max_iter=max_iter,
                        seed=base_seed,
                    )
                )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_repeat)(cfg, r) for r, cfg in enumerate(jobs)
    )

    # Flatten list of lists into a single DataFrame
    flat_rows: List[Dict] = [row for batch in results for row in batch]
    df = pd.DataFrame(flat_rows).sort_values(["T", "H", "iter", "repeat"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Bootstrap mean & CIs for each (T,H,iter,metric). Returns a tidy DataFrame:
    columns = [T, H, iter, algo, mean, lo, hi]; where algo in {"clustering","transition"}.
    """
    metrics = [c for c in df.columns if c not in ("T", "H", "repeat", "iter")]
    rows: List[Dict] = []
    for (T, H, it), g in df.groupby(["T", "H", "iter"], sort=True):
        for metric in metrics:
            vals = g[metric].to_numpy()
            means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
            rows.append({
                "T": T,
                "H": H,
                "iter": int(it),
                "algo": metric,
                "mean": float(means[0]),
                "lo": float(lo[0]),
                "hi": float(hi[0]),
            })
    out = pd.DataFrame(rows).sort_values(["T", "H", "iter", "algo"]).reset_index(drop=True)
    return out


# --------------------------- Plotting (usual CI panels) --------------------------- #

def plot_usual(summary: pd.DataFrame, savepath_prefix: str = "results/ablation2_results") -> None:
    # 1) Clustering error (CI line over iterations)
    def style_for_clustering(algo: str):
        return CB_PALETTE["clustering"], "-"

    plot_panels_with_ci(
        summary[summary["algo"].isin(["clustering", "oracle"])],
        series_order=["clustering", "oracle"],
        label_map={"clustering": "Clustering error", "oracle": "Oracle"},
        style_for=lambda a: (CB_PALETTE["clustering"], "-") if a == "clustering" else (CB_PALETTE["oracle"], ":"),
        savepath=f"{savepath_prefix}_clustering.png",
        facet_col="T",
        x_col="iter",
        series_col="algo",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=False,
        y_label="Error rate",
    )

    # 2) Transition matrix error (CI line over iterations)
    def style_for_transition(algo: str):
        return CB_PALETTE["transition"], "-"

    plot_panels_with_ci(
        summary[summary["algo"] == "transition"],
        series_order=["transition"],
        label_map={"transition": "Transition matrix error (L1)"},
        style_for=style_for_transition,
        savepath=f"{savepath_prefix}_transition.png",
        facet_col="T",
        x_col="iter",
        series_col="algo",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=False,
    )

    # 3) Total log-likelihood with oracle dotted line (green)
    def style_for_loglik(algo: str):
        return CB_PALETTE["loglik"], "-"

    sub_log = summary[summary["algo"].isin(["loglik"])]
    plot_panels_with_ci(
        sub_log,
        series_order=["loglik"],
        label_map={"loglik": "Total log-likelihood"},
        style_for=style_for_loglik,
        savepath=f"{savepath_prefix}_loglik.png",
        facet_col="T",
        x_col="iter",
        series_col="algo",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=False,
        y_label="Log likelihood",
    )


# --------------------------- Plotting (box-whisker by iteration) --------------------------- #

def plot_box_whiskers(df: pd.DataFrame, savepath_prefix: str = "results/ablation2_results") -> None:
    # Expect a single T and H; facet by T if multiple given
    unique_T = sorted(df["T"].unique())
    ncols = min(3, len(unique_T))
    nrows = int(np.ceil(len(unique_T) / ncols))

    # 1) Clustering error
    fig_c, axes_c = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.2*nrows), constrained_layout=True)
    axes_c = np.array(axes_c, ndmin=1).ravel()
    for i, T in enumerate(unique_T):
        ax = axes_c[i]
        sub = df[df["T"] == T]
        iters = sorted(sub["iter"].unique())
        data = [sub[sub["iter"] == it]["clustering"].to_numpy() for it in iters]
        bp = ax.boxplot(data, positions=iters, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(CB_PALETTE["clustering"])
            patch.set_alpha(0.3)
        for median in bp['medians']:
            median.set_color("#000000")
            median.set_linewidth(1.2)
        ax.set_title(f"T={T}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Clustering error rate")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax.set_xticks(iters)
    for j in range(len(unique_T), len(axes_c)):
        axes_c[j].axis("off")
    fig_c.suptitle("Clustering error by iteration", y=0.98)
    fig_c.savefig(f"{savepath_prefix}_clustering.png", dpi=300, bbox_inches="tight")
    plt.close(fig_c)

    # 2) Transition matrix error
    fig_t, axes_t = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.2*nrows), constrained_layout=True)
    axes_t = np.array(axes_t, ndmin=1).ravel()
    for i, T in enumerate(unique_T):
        ax = axes_t[i]
        sub = df[df["T"] == T]
        iters = sorted(sub["iter"].unique())
        data = [sub[sub["iter"] == it]["transition"].to_numpy() for it in iters]
        bp = ax.boxplot(data, positions=iters, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(CB_PALETTE["transition"])
            patch.set_alpha(0.3)
        for median in bp['medians']:
            median.set_color("#000000")
            median.set_linewidth(1.2)
        ax.set_title(f"T={T}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Transition matrix error (L1)")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax.set_xticks(iters)
    for j in range(len(unique_T), len(axes_t)):
        axes_t[j].axis("off")
    fig_t.suptitle("Transition matrix error by iteration", y=0.98)
    fig_t.savefig(f"{savepath_prefix}_transition.png", dpi=300, bbox_inches="tight")
    plt.close(fig_t)
    
    # 3) Total log-likelihood
    fig_l, axes_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.2*nrows), constrained_layout=True)
    axes_l = np.array(axes_l, ndmin=1).ravel()
    for i, T in enumerate(unique_T):
        ax = axes_l[i]
        sub = df[df["T"] == T]
        iters = sorted(sub["iter"].unique())
        data = [sub[sub["iter"] == it]["loglik"].to_numpy() for it in iters]
        bp = ax.boxplot(data, positions=iters, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor("#999999")
            patch.set_alpha(0.3)
        for median in bp['medians']:
            median.set_color("#000000")
            median.set_linewidth(1.2)
        ax.set_title(f"T={T}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total log-likelihood")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax.set_xticks(iters)
    for j in range(len(unique_T), len(axes_l)):
        axes_l[j].axis("off")
    fig_l.suptitle("Total log-likelihood by iteration", y=0.98)
    fig_l.savefig(f"{savepath_prefix}_loglik.png", dpi=300, bbox_inches="tight")

    plt.close(fig_l)

# --------------------------- CLI entry --------------------------- #

def main():
    # Keep the original defaults but use the unified structure
    T_list = [100, 200, 300, 400, 500, 600]
    H_list = [100]

    df = run_grid(
        T_list=T_list, H_list=H_list,
        K=3, S=10, delta=0.05,
        max_iter=10,
        n_repeat=30,
        n_jobs=-1,
        base_seed=2025,
    )
    df.to_csv("results/ablation2_results_raw.csv", index=False)

    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/ablation2_results_summary.csv", index=False)

    # Usual CI line plots (default)
    plot_usual(summary, savepath_prefix="results/ablation2_results")
    # Box-whisker plots remain available for exploratory analysis:
    # plot_box_whiskers(df, savepath_prefix="ablation2_results")
    # plot_box_whiskers(df, savepath_prefix="results/ablation2_results")

if __name__ == "__main__":
    main()



