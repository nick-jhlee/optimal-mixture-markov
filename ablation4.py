from __future__ import annotations

import numpy as np
import numpy.linalg as LA
import pandas as pd
from typing import Dict, List, Tuple

from Synthetic import MixtureMarkovChains
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from utils import error_rate, bootstrap_mean_ci_multi, KL
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def create_gamma_ps_varying_instance(
    q: float = 0.05,
    eps: float = 0.05,
    p1: float = 0.4,
    p2: float = 0.3,
) -> Dict:
    """
    MMC instance with K=2 and 10x10 transition matrices (Gemini construction).
    
    Key properties:
    - Rows 0,1 are FIXED (independent of q) and differ between P1 and P2 via p_k
    - Rows 2-9 depend on q but are IDENTICAL in P1 and P2
    - This ensures D_pi is approximately independent of q while gamma_ps varies with q
    
    Construction:
    - Row 0 for P_k: [1-p_k-ε, p_k, ε, 0, 0, 0, 0, 0, 0, 0]
    - Row 1 for P_k: [p_k, 1-p_k-ε, 0, 0, 0, 0, 0, 0, 0, ε]
    - Row 2 for both: [ε, 0, 1-q-ε, q, 0, 0, 0, 0, 0, 0]
    - Rows 3-8: chain structure with P[s,s-1]=q, P[s,s]=1-2q, P[s,s+1]=q
    - Row 9 for both: [0, ε, 0, 0, 0, 0, 0, 0, q, 1-q-ε]
    """
    S = 10
    
    # Validate parameters
    assert 0 < eps < 1 and 0 < p1 < 1 and 0 < p2 < 1, "eps, p1, p2 must be in (0, 1)"
    assert 0 < q < 0.5, "q must be in (0, 0.5) to keep 1-2q > 0"
    assert p1 + eps < 1 and p2 + eps < 1, "p_k + eps must be < 1"
    assert q + eps < 1, "q + eps must be < 1"
    
    # Initialize matrices
    P1 = np.zeros((S, S), dtype=float)
    P2 = np.zeros((S, S), dtype=float)
    
    # Row 0 for P1 and P2 (differs by p_k)
    P1[0] = [1-p1-eps, p1, eps, 0, 0, 0, 0, 0, 0, 0]
    P2[0] = [1-p2-eps, p2, eps, 0, 0, 0, 0, 0, 0, 0]
    
    # Row 1 for P1 and P2 (differs by p_k)
    P1[1] = [p1, 1-p1-eps, 0, 0, 0, 0, 0, 0, 0, eps]
    P2[1] = [p2, 1-p2-eps, 0, 0, 0, 0, 0, 0, 0, eps]
    
    # Row 2 for both (identical)
    P1[2] = [eps, 0, 1-q-eps, q, 0, 0, 0, 0, 0, 0]
    P2[2] = [eps, 0, 1-q-eps, q, 0, 0, 0, 0, 0, 0]
    
    # Rows 3-8 for both (identical chain structure)
    for s in range(3, 9):
        P1[s, s-1] = q
        P1[s, s] = 1 - 2*q
        P1[s, s+1] = q
        
        P2[s, s-1] = q
        P2[s, s] = 1 - 2*q
        P2[s, s+1] = q
    
    # Row 9 for both (identical)
    P1[9] = [0, eps, 0, 0, 0, 0, 0, 0, q, 1-q-eps]
    P2[9] = [0, eps, 0, 0, 0, 0, 0, 0, q, 1-q-eps]
    
    # Safety checks
    assert np.all(P1 >= 0) and np.all(P1 <= 1), "P1 has invalid probabilities"
    assert np.all(P2 >= 0) and np.all(P2 <= 1), "P2 has invalid probabilities"
    assert np.allclose(P1.sum(axis=1), 1.0), "P1 rows don't sum to 1"
    assert np.allclose(P2.sum(axis=1), 1.0), "P2 rows don't sum to 1"
    
    # Verify rows 2-9 are identical between P1 and P2
    for s in range(2, 10):
        assert np.allclose(P1[s], P2[s]), f"Row {s} should be identical in P1 and P2"
    
    # Use uniform initial distributions
    mu1 = np.full(S, 1.0/S)
    mu2 = np.full(S, 1.0/S)
    
    return {
        "S": S,
        "Ps": [P1, P2],
        "mus": [mu1, mu2],
    }


def compute_D_pi_manually(P1: np.ndarray, P2: np.ndarray) -> float:
    """
    Compute D_pi = min_{k≠k'} sum_s pi_k(s) KL(P_k(s,·) || P_k'(s,·))
    
    For this specific construction, D_pi should be independent of q.
    """
    S = P1.shape[0]
    
    # Compute stationary distributions
    def stationary(P):
        eigvals, eigvecs = LA.eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1))
        pi = np.real(eigvecs[:, idx])
        return pi / np.sum(pi)
    
    pi1 = stationary(P1)
    pi2 = stationary(P2)
    
    # Compute D_pi from chain 1 to chain 2
    D_pi_12 = 0.0
    for s in range(S):
        D_pi_12 += pi1[s] * KL(P1[s], P2[s])
    
    # Compute D_pi from chain 2 to chain 1
    D_pi_21 = 0.0
    for s in range(S):
        D_pi_21 += pi2[s] * KL(P2[s], P1[s])
    
    # Return the minimum (symmetric divergence concept)
    return min(D_pi_12, D_pi_21)


def run_one_repeat(
    T: int, H: int, q: float, delta: float, em_iters: int,
    repeat_idx: int, base_seed: int, eps: float, p1: float, p2: float
) -> Dict:
    """Run one experiment with specific q and H values."""
    if base_seed is not None:
        np.random.seed(base_seed + 37 * repeat_idx + int(1000 * q) + 7 * H)
    
    # Create instance with this q value
    gamma_config = create_gamma_ps_varying_instance(q=q, eps=eps, p1=p1, p2=p2)
    S = gamma_config["S"]
    K = 2
    
    env_config = {
        "H": H,
        "K": K,
        "S": S,
        "Ps": gamma_config["Ps"],
        "mus": gamma_config["mus"],
    }
    
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    D_pi = env.D_pi
    Delta_W = env.Delta_W
    
    # Also compute D_pi manually to verify it's independent of q
    D_pi_manual = compute_D_pi_manually(env.Ps[0], env.Ps[1])
    
    f_true, trajectories = env.generate_trajectories(T)
    true_arr = np.fromiter((f_true[t] for t in range(T)), dtype=int)
    
    # Oracle
    f_oracle = OracleLikelihoodRefinement(trajectories, env)
    pred_oracle = np.fromiter((f_oracle[t] for t in range(T)), dtype=int)
    err_oracle = error_rate(pred_oracle, true_arr)
    
    # Stage I (known K)
    f_hat_s1 = InitialSpectral(
        trajectories, T, H, S, gamma_ps, delta, K=K
    )
    pred_s1 = np.fromiter((f_hat_s1[t] for t in range(T)), dtype=int)
    err_s1 = error_rate(pred_s1, true_arr)
    
    # Stage I+II
    f_hat_s2 = LikelihoodRefinement(
        trajectories, f_hat_s1, T, S, max_iter=em_iters, history=False
    )
    pred_s2 = np.fromiter((f_hat_s2[t] for t in range(T)), dtype=int)
    err_s2 = error_rate(pred_s2, true_arr)
    
    return {
        "q": q,
        "T": T,
        "H": H,
        "repeat": repeat_idx,
        "gamma_ps": gamma_ps,
        "D_pi": D_pi,
        "D_pi_manual": D_pi_manual,
        "Delta_W": Delta_W,
        "oracle": err_oracle,
        "stage1": err_s1,
        "stage2": err_s2,
    }


def run_experiments(
    q_list: List[float],
    H_list: List[int],
    T: int = 200,
    delta: float = 0.05,
    em_iters: int = 10,
    eps: float = 0.05,
    p1: float = 0.4,
    p2: float = 0.3,
    n_repeat: int = 30,
    n_jobs: int = -1,
    base_seed: int = 2025,
) -> pd.DataFrame:
    """Run experiments varying H for each q value."""
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_repeat)(T, H, q, delta, em_iters, r, base_seed, eps, p1, p2)
        for q in q_list
        for H in H_list
        for r in range(n_repeat)
    )
    
    df = pd.DataFrame(results).sort_values(["q", "H", "repeat"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Bootstrap mean & CIs."""
    metrics = ["gamma_ps", "D_pi", "D_pi_manual", "Delta_W", "oracle", "stage1", "stage2"]
    rows = []
    
    for (q, H), g in df.groupby(["q", "H"], sort=True):
        for metric in metrics:
            vals = g[metric].to_numpy()
            means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
            rows.append({
                "q": q,
                "H": H,
                "metric": metric,
                "mean": float(means[0]),
                "lo": float(lo[0]),
                "hi": float(hi[0]),
            })
    
    out = pd.DataFrame(rows).sort_values(["q", "H", "metric"]).reset_index(drop=True)
    return out


def plot_results(summary: pd.DataFrame, savepath_prefix: str = "results/ablation4"):
    """
    Plot misclustering error vs H for several different gamma_ps values.
    Each gamma_ps corresponds to a fixed q value.
    """
    
    # Get unique q values and their corresponding gamma_ps
    q_values = summary["q"].unique()
    q_to_gamma = {}
    
    for q in q_values:
        gamma_mean = summary[(summary["q"] == q) & (summary["metric"] == "gamma_ps")]["mean"].values[0]
        q_to_gamma[q] = gamma_mean
    
    # Sort by gamma_ps for better legend ordering
    q_values_sorted = sorted(q_values, key=lambda q: q_to_gamma[q])
    
    # Print gamma_ps values for reference
    print(f"\nPlotting Error vs. H for {len(q_values_sorted)} different gamma_ps values:")
    print(f"{'q':>8s} | {'gamma_ps':>10s}")
    print("-" * 25)
    for q in q_values_sorted:
        print(f"{q:8.4f} | {q_to_gamma[q]:10.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_values_sorted)))
    
    # Plot each q (gamma_ps) as a separate curve
    for idx, q in enumerate(q_values_sorted):
        gamma_ps = q_to_gamma[q]
        
        # Get data for this q
        q_data = summary[(summary["q"] == q) & (summary["metric"] == "stage2")].sort_values("H")
        
        H_vals = q_data["H"].values
        error_mean = q_data["mean"].values
        error_lo = q_data["lo"].values
        error_hi = q_data["hi"].values
        
        # Plot
        ax.plot(H_vals, error_mean, 
                marker='o', color=colors[idx], linewidth=2.5, markersize=8,
                label=f"$\\gamma_{{\\mathrm{{ps}}}}$ = {gamma_ps:.4f}")
        ax.fill_between(H_vals, error_lo, error_hi, 
                       alpha=0.2, color=colors[idx])
    
    ax.set_xlabel("Horizon Length $H$", fontsize=20)
    ax.set_ylabel("Clustering Error", fontsize=20)
    ax.set_title(r"Misclustering Error vs. $H$ for Different $\gamma_{\mathrm{ps}}$", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f"{savepath_prefix}_error_vs_H.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{savepath_prefix}_error_vs_H.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {savepath_prefix}_error_vs_H.pdf and {savepath_prefix}_error_vs_H.png")


def main():
    print("=" * 70)
    print("GAMMA_PS DEPENDENCY ANALYSIS")
    print("=" * 70)
    
    # Parameters
    eps = 0.05
    p1 = 0.4
    p2 = 0.3
    
    # Display the matrix structure
    print("\nTransition Matrix Structure (10×10) - Gemini Construction:")
    print(f"Parameters: ε={eps}, p1={p1}, p2={p2}")
    print("\nP1(q):")
    print(f"  Row 0: [1-p1-ε, p1, ε, 0, ..., 0]  (fixed, depends on p1)")
    print(f"  Row 1: [p1, 1-p1-ε, 0, ..., 0, ε]  (fixed, depends on p1)")
    print("  Row 2: [ε, 0, 1-q-ε, q, 0, ..., 0]  (varies with q)")
    print("  Rows 3-8: chain [0, ..., q, 1-2q, q, ..., 0]  (varies with q)")
    print("  Row 9: [0, ε, 0, ..., 0, q, 1-q-ε]  (varies with q)")
    print("\nP2(q):")
    print(f"  Row 0: [1-p2-ε, p2, ε, 0, ..., 0]  (fixed, depends on p2)")
    print(f"  Row 1: [p2, 1-p2-ε, 0, ..., 0, ε]  (fixed, depends on p2)")
    print("  Rows 2-9: IDENTICAL to P1  (varies with q)")
    print("\nKey: Rows 0,1 differ between chains (fixed D_pi), rows 2-9 control gamma_ps via q")
    
    # Sample instance to show D_pi is independent of q
    print("\n" + "=" * 70)
    print("VERIFYING D_pi INDEPENDENCE FROM q:")
    print("=" * 70)
    
    q_samples = [0.3, 0.2, 0.1, 0.05, 0.01]
    print(f"{'q':>6s} | {'gamma_ps':>10s} | {'D_pi':>10s}")
    print("-" * 35)
    
    for q in q_samples:
        config = create_gamma_ps_varying_instance(q=q, eps=eps, p1=p1, p2=p2)
        P1, P2 = config["Ps"]
        D_pi = compute_D_pi_manually(P1, P2)
        
        # Quick gamma_ps estimate (not exact, but illustrative)
        env_config = {"H": 100, "K": 2, "S": 10, "Ps": config["Ps"], "mus": config["mus"]}
        env = MixtureMarkovChains(env_config)
        gamma_ps = env.gamma_ps
        
        print(f"{q:6.2f} | {gamma_ps:10.6f} | {D_pi:10.6f}")
    
    print("\nObservation: D_pi should be approximately constant across different q values.")
    print("=" * 70)
    
    # Select a few q values to span different gamma_ps regimes
    # q controls mixing in rows 2-9 while rows 0,1 create fixed divergence
    q_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2]  # Will give different gamma_ps values
    
    # Vary H to see how performance improves with more data
    H_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    
    print("\nRunning full experiments varying H for each q...")
    print(f"q values: {q_list}")
    print(f"H values: {H_list}")
    print("=" * 70)
    
    df = run_experiments(
        q_list=q_list,
        H_list=H_list,
        T=200,
        delta=0.05,
        em_iters=10,
        eps=eps,
        p1=p1,
        p2=p2,
        n_repeat=30,
        n_jobs=-1,
        base_seed=2025,
    )
    
    df.to_csv("results/ablation4_error_vs_H_raw.csv", index=False)
    
    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/ablation4_error_vs_H_summary.csv", index=False)
    
    # Verify D_pi constancy
    dpi_summary = summary[summary["metric"] == "D_pi_manual"].drop_duplicates(subset=["q"])
    print("\n" + "=" * 70)
    print("D_pi VALUES FOR EACH q:")
    print(f"{'q':>8s} | {'D_pi':>10s}")
    print("-" * 25)
    for _, row in dpi_summary.sort_values("q").iterrows():
        print(f"{row['q']:8.4f} | {row['mean']:10.6f}")
    
    print(f"\nD_pi Statistics:")
    print(f"  Mean:   {dpi_summary['mean'].mean():.6f}")
    print(f"  Std:    {dpi_summary['mean'].std():.6f}")
    print(f"  Range:  {dpi_summary['mean'].max() - dpi_summary['mean'].min():.6f}")
    print("  => D_pi is approximately constant (as expected)")
    print("=" * 70)
    
    plot_results(summary, savepath_prefix="results/ablation4")
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("  Raw data: results/ablation4_error_vs_H_raw.csv")
    print("  Summary:  results/ablation4_error_vs_H_summary.csv")
    print("  Plots:    results/ablation4_error_vs_H.{pdf,png}")
    print("=" * 70)


if __name__ == "__main__":
    main()
