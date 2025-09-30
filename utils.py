from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

ArrayLike = Union[np.ndarray, List[int]]
LabelLike = Union[Dict[int, int], ArrayLike]

# ------------------------------
# 1) Optimal error rate (Hungarian)
# ------------------------------

def _to_label_array(labels: LabelLike) -> np.ndarray:
    """Accept dict (t->label) or 1D array/list; return int32 array of shape (T,)."""
    if isinstance(labels, dict):
        T = len(labels)
        if sorted(labels.keys()) != list(range(T)):
            # allow arbitrary keys but keep consistent order by sorted key
            items = sorted(labels.items())
            arr = np.fromiter((lab for _, lab in items), dtype=np.int32, count=T)
        else:
            arr = np.fromiter((labels[t] for t in range(T)), dtype=np.int32, count=T)
        return arr
    arr = np.asarray(labels, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("Labels must be 1D.")
    return arr

def _remap_to_compact(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remap arbitrary integer labels to {0,...,K-1}. Returns (remapped, inverse_map)."""
    uniq = np.unique(labels)
    inv = -np.ones(uniq.max() + 1, dtype=np.int32)
    inv[uniq] = np.arange(uniq.size, dtype=np.int32)
    return inv[labels], uniq  # uniq[g_remapped] gives original label

def error_rate(
    f: LabelLike, g: LabelLike, return_perm: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Clustering error rate with optimal permutation (Hungarian).
    - Accepts dicts or arrays.
    - Handles non-contiguous / arbitrary integer labels.
    - Time: O(T + K^2 + K^3) ~ dominated by Hungarian's O(K^3), fine for typical K.
    """
    f_arr = _to_label_array(f)
    g_arr = _to_label_array(g)
    if f_arr.shape != g_arr.shape:
        raise ValueError("Mismatch in lengths")

    T = f_arr.size
    if T == 0:
        return (0.0, np.array([], dtype=int)) if return_perm else 0.0

    f_c, f_vals = _remap_to_compact(f_arr)
    g_c, g_vals = _remap_to_compact(g_arr)
    K = max(f_c.max(initial=-1), g_c.max(initial=-1)) + 1
    if K == 0:
        return (0.0, np.array([], dtype=int)) if return_perm else 0.0

    # Confusion matrix via bincount: C[i,j] = # {t: f_c[t]=i, g_c[t]=j}
    lin = f_c * K + g_c
    C = np.bincount(lin, minlength=K*K).reshape(K, K)

    # We want to maximize matches => minimize -C
    row_ind, col_ind = linear_sum_assignment(-C)
    matches = C[row_ind, col_ind].sum()
    err = 1.0 - (matches / T)

    if return_perm:
        # Map canonical f-label i to g-label col_ind[i] (in canonical space),
        # then convert back to original g labels.
        perm_in_g_space = g_vals[col_ind]
        # Provide a size-K array: f_vals[i] -> perm_in_g_space[i]
        # (If consumer expects canonical, they can use col_ind.)
        return err, perm_in_g_space
    return err

# ------------------------------
# 2) Vectorized log-likelihood for a trajectory
# ------------------------------

def log_likelihood(
    trajectory: ArrayLike, P: np.ndarray, mu: Optional[np.ndarray] = None, eps: float = 1e-300
) -> float:
    """
    log P(traj) = log mu[s0] + sum_{t} log P[s_t, s_{t+1}]
    Uses clipping to avoid log(0); set eps=0.0 to allow -inf if desired.
    """
    traj = np.asarray(trajectory, dtype=np.int64)
    if traj.ndim != 1:
        raise ValueError("trajectory must be 1D")
    if traj.size == 0:
        return 0.0

    if mu is not None:
        ll = float(np.log(np.clip(mu[traj[0]], eps, 1.0)))
    else:
        ll = 0.0

    steps = P[traj[:-1], traj[1:]]
    ll += float(np.log(np.clip(steps, eps, 1.0)).sum())
    return ll

# ------------------------------
# 3) k-step empirical visitation matrix (stride=1)
# ------------------------------

def empirical_visitation_matrix(
    trajectory: List[int], S: int, k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      N: (S,S) int64 matrix with counts of k-step transitions
      N_row: (S,) int64 vector with row sums
    """
    traj = np.asarray(trajectory, dtype=np.int64)
    H = traj.size
    if k < 1:
        raise ValueError("k must be >= 1")
    if H <= k:
        return np.zeros((S, S), dtype=np.int64), np.zeros(S, dtype=np.int64)
    if (traj < 0).any() or (traj >= S).any():
        raise ValueError("trajectory contains state outside [0, S)")

    s = traj[:-k]
    t = traj[k:]
    idx = s * S + t
    N = np.bincount(idx, minlength=S*S).reshape(S, S).astype(np.int64)
    N_row = N.sum(axis=1)
    return N, N_row

# ------------------------------
# 4) Clustered transition matrix with smoothing
# ------------------------------

def clustered_transition_matrix(
    trajectories: List[List[int]],
    indices: List[int],
    S: int,
    *,
    k: int = 1,
    alpha: float = 1.0  # Dirichlet smoothing
) -> np.ndarray:
    """
    Estimate cluster-level transition matrix with Dirichlet(alpha) smoothing.
    If indices is empty: return uniform rows.
    """
    if len(indices) == 0:
        return np.full((S, S), 1.0 / S, dtype=float)

    N = np.zeros((S, S), dtype=np.int64)
    N_row = np.zeros(S, dtype=np.int64)
    for t_idx in indices:
        N_t, N_row_t = empirical_visitation_matrix(trajectories[t_idx], S, k=k)
        N += N_t
        N_row += N_row_t

    # Dirichlet smoothing: add alpha pseudo-counts per outgoing edge
    N_sm = N.astype(float) + alpha
    denom = (N_row.astype(float) + alpha * S).reshape(-1, 1)
    P_hat = N_sm / np.clip(denom, 1e-300, None)
    return P_hat

# ------------------------------
# 5) Transition matrix error (L1)
# ------------------------------
def transition_matrix_error(
    P_estimated: np.ndarray,
    P_true: np.ndarray,
    *,
    pad: str = "uniform",
    return_perm: bool = False
):
    """
    Minimum total entrywise L1 error between sets of transition matrices
    after optimally permuting clusters (Hungarian assignment).

    Args:
        P_estimated: (K_est, S, S) estimated transition matrices
        P_true:      (K_true, S, S) true transition matrices
        pad:         how to pad when K_est != K_true: "uniform" or "zeros"
        return_perm: if True, also return the permutation mapping estimated->true indices

    Returns:
        total_l1_error (float)  -- sum_{k} || P_estimated[perm[k]] - P_true[k] ||_1
        (optional) perm (np.ndarray of shape (K*,), where K* = max(K_est, K_true))
    """
    P_estimated = np.asarray(P_estimated, dtype=float)
    P_true      = np.asarray(P_true, dtype=float)

    if P_estimated.ndim != 3 or P_true.ndim != 3:
        raise ValueError("P_estimated and P_true must be 3D arrays of shape (K, S, S).")

    K_est, S1, S2 = P_estimated.shape
    K_true, S1_, S2_ = P_true.shape
    if (S1 != S2) or (S1_ != S2_) or (S1 != S1_):
        raise ValueError("Both inputs must have shape (K, S, S) with the same S.")

    S = S1
    K = max(K_est, K_true)

    # Pad the smaller set to size K
    if K_est < K:
        if pad == "uniform":
            fill = np.full((K - K_est, S, S), 1.0 / S)
        elif pad == "zeros":
            fill = np.zeros((K - K_est, S, S))
        else:
            raise ValueError("pad must be 'uniform' or 'zeros'.")
        P_est = np.concatenate([P_estimated, fill], axis=0)
        P_ref = P_true
    elif K_true < K:
        if pad == "uniform":
            fill = np.full((K - K_true, S, S), 1.0 / S)
        elif pad == "zeros":
            fill = np.zeros((K - K_true, S, S))
        else:
            raise ValueError("pad must be 'uniform' or 'zeros'.")
        P_est = P_estimated
        P_ref = np.concatenate([P_true, fill], axis=0)
    else:
        P_est = P_estimated
        P_ref = P_true

    # Cost matrix D[i,j] = L1 distance between P_est[i] and P_ref[j]
    # Vectorized: broadcast (K,S,S) vs (K,S,S) -> (K,K,S,S), then sum abs over (S,S)
    D = np.abs(P_est[:, None, :, :] - P_ref[None, :, :, :]).sum(axis=(2, 3))

    # Hungarian assignment to minimize total L1 error
    row_ind, col_ind = linear_sum_assignment(D)
    total_error = float(D[row_ind, col_ind].sum())

    if return_perm:
        # perm[i] = matched index in P_ref for P_est[i]
        perm = np.full(K, -1, dtype=int)
        perm[row_ind] = col_ind
        return total_error, perm
    return total_error


# ------------------------------
# 6) KL divergence (robust)
# ------------------------------

def KL(p: np.ndarray, q: np.ndarray) -> float:
    """
    D(p||q) = sum_i p_i log(p_i/q_i)
    - ignores entries where p_i=0
    - returns +inf if q_i=0 while p_i>0
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")
    mask = p > 0
    if not np.any(mask):
        return 0.0
    if np.any(q[mask] == 0.0):
        return float('inf')
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

# ------------------------------
# 7) Bootstrap paired CI for multiple algorithms
# ------------------------------

def bootstrap_mean_ci_multi(
    scalar_matrix: ArrayLike, alpha: float = 0.05, n_bootstrap: int = 5000, seed: Optional[int] = None
):
    """
    Percentile CIs for the mean of K algorithms measured on the same n runs.
    Returns (mean_vec, ci_lower, ci_upper), each shape (K,).
    """
    X = np.asarray(scalar_matrix, dtype=float)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if X.size == 0:
        raise ValueError("No finite rows.")
    n, k = X.shape
    rng = np.random.default_rng(seed)

    mean_vec = X.mean(axis=0)

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = X[idx].mean(axis=1)  # (B, K)

    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    ci_lower = np.quantile(boot_means, lo, axis=0)
    ci_upper = np.quantile(boot_means, hi, axis=0)
    return mean_vec, ci_lower, ci_upper
