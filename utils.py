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