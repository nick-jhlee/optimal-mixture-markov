from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import permutations
from typing import Dict

# Constants for optimization
HUNGARIAN_THRESHOLD = 5  # Use Hungarian algorithm for K >= this value

def error_rate(f: Dict[int, int], g: Dict[int, int]) -> float:
    # f and g are dictionaries that map [T] to [K]
    T = len(f.keys())
    if T != len(g.keys()):
        raise ValueError("Mismatch in T")
    # K = len(set(f.values()))
    # if K != len(set(g.values())):
    #     raise ValueError("Mismatch in K")
    K = max(len(f.values()), len(g.values()))
    
    # Convert to arrays for vectorized computation
    f_arr = np.array([f[t] for t in range(T)], dtype=np.int32)
    g_arr = np.array([g[t] for t in range(T)], dtype=np.int32)
    
    # Use Hungarian algorithm for optimal assignment (much faster than brute force)
    if K >= HUNGARIAN_THRESHOLD:  # Use Hungarian for large K
        # Create cost matrix
        cost_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost_matrix[i, j] = np.sum((f_arr == i) & (g_arr != j))
        
        _, assignment = linear_sum_assignment(cost_matrix)
        error = cost_matrix[np.arange(K), assignment].sum()
        return error / T
    else:
        # Fallback to original brute force for reasonable K
        error = T
        for perm_K in permutations(range(K)):
            new_error = sum([perm_K[f[t]] != g[t] for t in range(T)])
            if error > new_error:
                error = new_error
        return error / T
    

def KL(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D(p||q) = sum_i p_i * log(p_i / q_i)
    Returns -∞ if there's division by zero (q has zeros where p doesn't)"""
    
    # Check for division by zero: if q has zeros where p doesn't
    if np.any((q == 0) & (p > 0)):
        return float('-inf')
    
    # Handle p = 0 case: 0 * log(0/q) = 0
    mask = p > 0
    if not np.any(mask):
        return 0.0
    
    # Compute KL only where p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def bootstrap_mean_ci_multi(scalar_matrix, alpha=0.05, n_bootstrap=5000, seed=None):
    """
    Percentile bootstrap CIs for the mean of K algorithms measured on the same n runs.
    scalar_matrix: array-like of shape (n_runs, n_algs)
      Each column is an algorithm; each row is a run/trajectory (shared across algs).
    Returns:
      mean_vec: (n_algs,)
      ci_lower: (n_algs,)
      ci_upper: (n_algs,)
    """
    X = np.asarray(scalar_matrix, dtype=float)
    # keep only rows where all algs are finite (preserve pairing)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if X.size == 0:
        raise ValueError("No finite rows.")

    rng = np.random.default_rng(seed)
    n, k = X.shape

    mean_vec = X.mean(axis=0)

    # paired resampling of rows
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = X[idx].mean(axis=1)   # shape: (B, k)

    lower_q, upper_q = alpha/2, 1 - alpha/2
    ci_lower = np.quantile(boot_means, lower_q, axis=0)
    ci_upper = np.quantile(boot_means, upper_q, axis=0)
    return mean_vec, ci_lower, ci_upper