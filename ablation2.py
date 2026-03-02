"""
Ablation #2. Impact of S and T on Performance and Runtime

This script analyzes the impact of state-space size S and number of trajectories T on the performance and runtime of our two-stage algorithm.

Reproducibility:
    All experiments use deterministic seeding for full reproducibility:
    - Base seed (2025) is set globally at the start of main()
    - Each experiment gets a unique seed = base_seed + hash(repeat_idx, T, H, S)
    - This ensures consistent results across runs while maintaining independence
    
    To reproduce results: simply run this script with the same base seed.
    
Output:
    - Raw data: results/ablation2_runtime_vs_{S,T}_raw.csv
    - Summary statistics: results/ablation2_runtime_summary.csv
    - Plots: results/ablation2_runtime.{pdf,png}
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.linalg as LA
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import L_embedding, LikelihoodRefinement
from utils import bootstrap_mean_ci_multi, error_rate, empirical_visitation_matrix, clustered_transition_matrix, log_likelihood
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: We also set per-experiment seeds in run_one_repeat for parallel safety


@dataclass(frozen=True)
class RuntimeConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    em_iters: int
    seed: int | None = None


@dataclass
class TimingResult:
    embedding_time: float
    svd_time: float
    clustering_time: float
    refinement_time: float
    total_time: float
    error_stage1: float
    error_stage2: float


def timed_initial_spectral(
    trajectories: List[List[int]], 
    T: int, 
    H: int, 
    S: int, 
    K: int
) -> Tuple[Dict[int, int], float, float, float]:
    """Run InitialSpectral with timing instrumentation."""
    
    # Time embedding construction
    t0 = time.perf_counter()
    W_hat = L_embedding(trajectories, T, H, S)
    t1 = time.perf_counter()
    embedding_time = t1 - t0
    
    # Time SVD
    t0 = time.perf_counter()
    U, Sigma, _ = LA.svd(W_hat)
    t1 = time.perf_counter()
    svd_time = t1 - t0
    
    # Time clustering (K-means)
    t0 = time.perf_counter()
    X_hat = U[:, :K] @ np.diag(Sigma[:K])
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X_hat)
    f_hat = {t: kmeans.labels_[t] for t in range(T)}
    t1 = time.perf_counter()
    clustering_time = t1 - t0
    
    return f_hat, embedding_time, svd_time, clustering_time


def run_one_repeat(cfg: RuntimeConfig, repeat_idx: int) -> Dict:
    """
    Run a single experiment and measure timing breakdown.
    
    Seeding strategy for reproducibility:
    - Each experiment gets a unique, deterministic seed
    - Seed = base_seed + hash(repeat_idx, T, H, S)
    - This ensures: (1) reproducible results, (2) independence across trials
    """
    if cfg.seed is not None:
        # Unique seed per (repeat, T, H, S) combination
        np.random.seed(cfg.seed + 37 * repeat_idx + 1009 * cfg.T + 17 * cfg.H + 23 * cfg.S)
    
    # Generate environment and trajectories
    env_config = _build_env_config(cfg.H, cfg.K, cfg.S)
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    f_true, trajectories = env.generate_trajectories(cfg.T)
    true_arr = np.fromiter((f_true[t] for t in range(cfg.T)), dtype=int)
    
    # Stage I with timing
    f_hat_s1, emb_time, svd_time, clust_time = timed_initial_spectral(
        trajectories, cfg.T, cfg.H, cfg.S, cfg.K
    )
    pred_s1 = np.fromiter((f_hat_s1[t] for t in range(cfg.T)), dtype=int)
    err_s1 = error_rate(pred_s1, true_arr)
    
    # Stage II (EM refinement) with timing
    t0 = time.perf_counter()
    f_hat_s2 = LikelihoodRefinement(
        trajectories, f_hat_s1, cfg.T, cfg.S, max_iter=cfg.em_iters, history=False
    )
    t1 = time.perf_counter()
    refinement_time = t1 - t0
    
    pred_s2 = np.fromiter((f_hat_s2[t] for t in range(cfg.T)), dtype=int)
    err_s2 = error_rate(pred_s2, true_arr)
    
    total_time = emb_time + svd_time + clust_time + refinement_time
    
    return {
        "T": cfg.T,
        "H": cfg.H,
        "S": cfg.S,
        "repeat": repeat_idx,
        "embedding_time": emb_time,
        "svd_time": svd_time,
        "clustering_time": clust_time,
        "refinement_time": refinement_time,
        "total_time": total_time,
        "error_stage1": err_s1,
        "error_stage2": err_s2,
    }


def run_grid(
    T_list: List[int],
    H_list: List[int],
    S_list: List[int],
    *,
    K: int = 3,
    delta: float = 0.05,
    em_iters: int = 10,
    n_repeat: int = 30,
    n_jobs: int = -1,
    base_seed: int | None = 2025,
) -> pd.DataFrame:
    """Run all (T, H, S) combinations × repeats."""
    jobs: List[RuntimeConfig] = []
    for T in T_list:
        for H in H_list:
            for S in S_list:
                for r in range(n_repeat):
                    jobs.append(
                        RuntimeConfig(
                            T=T, H=H, K=K, S=S, delta=delta,
                            em_iters=em_iters, seed=base_seed,
                        )
                    )
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_repeat)(cfg, r) for r, cfg in enumerate(jobs)
    )
    df = pd.DataFrame(results).sort_values(["T", "H", "S", "repeat"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Bootstrap mean & CIs for each configuration."""
    metrics = [
        "embedding_time", "svd_time", "clustering_time", 
        "refinement_time", "total_time", "error_stage1", "error_stage2"
    ]
    rows = []
    
    for group_keys, g in df.groupby(["T", "H", "S"], sort=True):
        for metric in metrics:
            vals = g[metric].to_numpy()
            means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
            row = {"T": group_keys[0], "H": group_keys[1], "S": group_keys[2],
                   "metric": metric, "mean": float(means[0]),
                   "lo": float(lo[0]), "hi": float(hi[0])}
            rows.append(row)
    
    return pd.DataFrame(rows).sort_values(["T", "H", "S", "metric"]).reset_index(drop=True)


def plot_runtime_analysis(summary: pd.DataFrame, savepath_prefix: str = "results/ablation2"):
    """
    Create 6 plots in 3 rows:
    Row 1: Performance (error) vs S and T
    Row 2: Component-wise runtime vs S and T
    Row 3: Total runtime vs S (with quadratic fit) and T (with linear fit)
    """
    
    # Define colors for timing components
    colors = {
        "embedding_time": "#E69F00",
        "svd_time": "#56B4E9",
        "clustering_time": "#009E73",
        "refinement_time": "#D55E00",
        "total_time": "#000000",
    }
    
    labels = {
        "embedding_time": "Embedding",
        "svd_time": "SVD",
        "clustering_time": "Clustering",
        "refinement_time": "Likelihood Refinement",
        "total_time": "Total",
    }
    
    # Find fixed values for each dimension
    s_counts = summary.groupby("T")["S"].nunique()
    T_fixed = s_counts.idxmax()  # T that has most S variations
    t_counts = summary.groupby("S")["T"].nunique()
    S_fixed = t_counts.idxmax()  # S that has most T variations
    H_fixed = summary["H"].unique()[0]
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))  # Increased from 14x16 to 16x20
    
    # ==================== ROW 0: PERFORMANCE ====================
    
    # --- Plot 0,0: Performance vs S ---
    ax = axes[0, 0]
    subset = summary[(summary["T"] == T_fixed) & (summary["H"] == H_fixed)]
    
    for metric in ["error_stage1", "error_stage2"]:
        metric_data = subset[subset["metric"] == metric].sort_values("S")
        label = "Stage I" if metric == "error_stage1" else "Stage I+II"
        linestyle = "--" if metric == "error_stage1" else "-"
        ax.plot(metric_data["S"], metric_data["mean"], 
                marker='o', label=label, linestyle=linestyle, linewidth=2)
        ax.fill_between(metric_data["S"], metric_data["lo"], metric_data["hi"], alpha=0.2)
    
    ax.set_xlabel("S (State space size)", fontsize=18)
    ax.set_ylabel("Clustering Error", fontsize=18)
    ax.set_title(f"Performance vs S (T={T_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 0,1: Performance vs T ---
    ax = axes[0, 1]
    subset = summary[(summary["S"] == S_fixed) & (summary["H"] == H_fixed)]
    
    for metric in ["error_stage1", "error_stage2"]:
        metric_data = subset[subset["metric"] == metric].sort_values("T")
        label = "Stage I" if metric == "error_stage1" else "Stage I+II"
        linestyle = "--" if metric == "error_stage1" else "-"
        ax.plot(metric_data["T"], metric_data["mean"], 
                marker='o', label=label, linestyle=linestyle, linewidth=2)
        ax.fill_between(metric_data["T"], metric_data["lo"], metric_data["hi"], alpha=0.2)
    
    ax.set_xlabel("T (Number of trajectories)", fontsize=18)
    ax.set_ylabel("Clustering Error", fontsize=18)
    ax.set_title(f"Performance vs T (S={S_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # ==================== ROW 1: COMPONENT-WISE RUNTIME ====================
    
    # --- Plot 1,0: Runtime vs S ---
    ax = axes[1, 0]
    subset = summary[(summary["T"] == T_fixed) & (summary["H"] == H_fixed)]
    
    timing_metrics = ["embedding_time", "svd_time", "clustering_time", "refinement_time"]
    for metric in timing_metrics:
        metric_data = subset[subset["metric"] == metric].sort_values("S")
        ax.plot(metric_data["S"], metric_data["mean"], 
                marker='o', label=labels[metric], color=colors[metric], linewidth=2)
        ax.fill_between(metric_data["S"], metric_data["lo"], metric_data["hi"], 
                        alpha=0.2, color=colors[metric])
    
    ax.set_xlabel("S (State space size)", fontsize=18)
    ax.set_ylabel("Time (seconds)", fontsize=18)
    ax.set_title(f"Component Runtime vs S (T={T_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # --- Plot 1,1: Runtime vs T ---
    ax = axes[1, 1]
    subset = summary[(summary["S"] == S_fixed) & (summary["H"] == H_fixed)]
    
    for metric in timing_metrics:
        metric_data = subset[subset["metric"] == metric].sort_values("T")
        ax.plot(metric_data["T"], metric_data["mean"], 
                marker='o', label=labels[metric], color=colors[metric], linewidth=2)
        ax.fill_between(metric_data["T"], metric_data["lo"], metric_data["hi"], 
                        alpha=0.2, color=colors[metric])
    
    ax.set_xlabel("T (Number of trajectories)", fontsize=18)
    ax.set_ylabel("Time (seconds)", fontsize=18)
    ax.set_title(f"Component Runtime vs T (S={S_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # ==================== ROW 2: TOTAL RUNTIME WITH FITTING ====================
    
    # --- Plot 2,0: Total Time vs S (Quadratic fit) ---
    ax = axes[2, 0]
    subset = summary[(summary["T"] == T_fixed) & (summary["H"] == H_fixed)]
    total_data = subset[subset["metric"] == "total_time"].sort_values("S")
    
    S_vals = total_data["S"].values
    time_vals = total_data["mean"].values
    
    # Fit quadratic: f(S) = a*S^2 + b*S + c
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    popt_quad, _ = curve_fit(quadratic, S_vals, time_vals)
    a_fit, b_fit, c_fit = popt_quad
    
    # Compute R²
    y_pred = quadratic(S_vals, *popt_quad)
    ss_res = np.sum((time_vals - y_pred)**2)
    ss_tot = np.sum((time_vals - np.mean(time_vals))**2)
    r2_quad = 1 - (ss_res / ss_tot)
    
    # Plot data and fit
    ax.plot(S_vals, time_vals, 'o', color=colors["total_time"], 
            markersize=10, label='Data', zorder=3)
    ax.fill_between(total_data["S"], total_data["lo"], total_data["hi"], 
                    alpha=0.2, color=colors["total_time"])
    
    S_fine = np.linspace(S_vals.min(), S_vals.max(), 100)
    ax.plot(S_fine, quadratic(S_fine, *popt_quad), '--', 
            color='red', linewidth=2, label=f'Quadratic fit: R²={r2_quad:.4f}', zorder=2)
    
    ax.set_xlabel("S (State space size)", fontsize=18)
    ax.set_ylabel("Total Time (seconds)", fontsize=18)
    ax.set_title(f"Total Runtime vs S (T={T_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # # Add text box with fit equation
    # textstr = f'$f(S) = {a_fit:.2e}S^2 + {b_fit:.2e}S + {c_fit:.2e}$\n$R^2 = {r2_quad:.4f}$'
    # ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='bottom', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Add text box with R² only
    textstr = f'$R^2 = {r2_quad:.4f}$'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    # --- Plot 2,1: Total Time vs T (Linear fit) ---
    ax = axes[2, 1]
    subset = summary[(summary["S"] == S_fixed) & (summary["H"] == H_fixed)]
    total_data = subset[subset["metric"] == "total_time"].sort_values("T")
    
    T_vals = total_data["T"].values
    time_vals = total_data["mean"].values
    
    # Fit linear: f(T) = a*T + b
    def linear(x, a, b):
        return a * x + b
    
    popt_lin, _ = curve_fit(linear, T_vals, time_vals)
    a_lin, b_lin = popt_lin
    
    # Compute R²
    y_pred = linear(T_vals, *popt_lin)
    ss_res = np.sum((time_vals - y_pred)**2)
    ss_tot = np.sum((time_vals - np.mean(time_vals))**2)
    r2_lin = 1 - (ss_res / ss_tot)
    
    # Compute Pearson correlation
    corr, p_value = pearsonr(T_vals, time_vals)
    
    # Plot data and fit
    ax.plot(T_vals, time_vals, 'o', color=colors["total_time"], 
            markersize=10, label='Data', zorder=3)
    ax.fill_between(total_data["T"], total_data["lo"], total_data["hi"], 
                    alpha=0.2, color=colors["total_time"])
    
    T_fine = np.linspace(T_vals.min(), T_vals.max(), 100)
    ax.plot(T_fine, linear(T_fine, *popt_lin), '--', 
            color='blue', linewidth=2, label=f'Linear fit: R²={r2_lin:.4f}', zorder=2)
    
    ax.set_xlabel("T (Number of trajectories)", fontsize=18)
    ax.set_ylabel("Total Time (seconds)", fontsize=18)
    ax.set_title(f"Total Runtime vs T (S={S_fixed}, H={H_fixed})", fontsize=20)
    ax.legend(fontsize=15, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # # Add text box with fit equation
    # textstr = f'$f(T) = {a_lin:.2e}T + {b_lin:.2e}$\n$R^2 = {r2_lin:.4f}$\n$r = {corr:.4f}$, $p < {p_value:.2e}$'
    # ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='bottom', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    # Add text box with R² only
    textstr = f'$R^2 = {r2_lin:.4f}$'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{savepath_prefix}.png", dpi=300, bbox_inches='tight')
    
    # Print statistics
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS OF RUNTIME SCALING")
    print(f"{'='*70}")
    print(f"\nQuadratic fit for Total Runtime vs S (T={T_fixed}, H={H_fixed}):")
    print(f"  f(S) = {a_fit:.6e} * S² + {b_fit:.6e} * S + {c_fit:.6e}")
    print(f"  R² = {r2_quad:.6f}")
    print(f"\nLinear fit for Total Runtime vs T (S={S_fixed}, H={H_fixed}):")
    print(f"  f(T) = {a_lin:.6e} * T + {b_lin:.6e}")
    print(f"  R² = {r2_lin:.6f}")
    print(f"  Pearson r = {corr:.6f}, p-value = {p_value:.2e}")
    print(f"{'='*70}\n")
    
    print(f"Saved plots to {savepath_prefix}.pdf and {savepath_prefix}.png")


def main():
    """
    Main function for runtime analysis experiments.
    
    Reproducibility notes:
    - Base seed is set globally at the start
    - Each experiment gets a deterministic seed based on: base_seed + repeat_idx + T + H + S
    - This ensures consistent results across runs while maintaining independence across trials
    """
    # Set global seed for reproducibility
    BASE_SEED = 2025
    set_seed(BASE_SEED)
    
    print(f"Running experiments with base seed: {BASE_SEED}")
    print("Note: Each experiment uses a deterministic seed derived from the base seed")
    print("=" * 70)
    
    # Vary S: keep T and H fixed
    T_list = [400]
    H_list = [500]
    S_list = [10, 20, 30, 40, 50, 60]
    
    print("\nRunning experiments varying S...")
    df_S = run_grid(
        T_list=T_list,
        H_list=H_list,
        S_list=S_list,
        K=3,
        delta=0.05,
        em_iters=10,
        n_repeat=30,
        n_jobs=-1,
        base_seed=BASE_SEED,
    )
    df_S.to_csv("results/ablation2_runtime_vs_S_raw.csv", index=False)
    
    # Vary T: keep S and H fixed
    T_list = [100, 200, 300, 400, 500, 600]
    H_list = [500]
    S_list = [40]
    
    print("\nRunning experiments varying T...")
    df_T = run_grid(
        T_list=T_list,
        H_list=H_list,
        S_list=S_list,
        K=3,
        delta=0.05,
        em_iters=10,
        n_repeat=30,
        n_jobs=-1,
        base_seed=BASE_SEED,
    )
    df_T.to_csv("results/ablation2_runtime_vs_T_raw.csv", index=False)
    
    # Combine and summarize
    df_combined = pd.concat([df_S, df_T], ignore_index=True)
    summary = summarize(df_combined, alpha=0.05)
    summary.to_csv("results/ablation2_runtime_summary.csv", index=False)
    
    # Plot
    plot_runtime_analysis(summary, savepath_prefix="results/ablation2_runtime")
    
    print("\nRuntime analysis complete!")
    print("Results saved to results/ablation2_runtime_*.csv")
    print("Plots saved to results/ablation2_runtime.{pdf,png}")


if __name__ == "__main__":
    main()
