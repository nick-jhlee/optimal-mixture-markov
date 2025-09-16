import numpy as np
import numpy.linalg as LA
from typing import List, Tuple, Dict

# Constants for optimization
SIGMA_THRESHOLD_FACTOR = 5e-2  # Factor for sigma threshold calculation
RHO_THRESHOLD_FACTOR = 1e-1    # Factor for rho threshold calculation
LOG_EPSILON = 1e-10            # Small epsilon to avoid log(0)

def empirical_transition_matrix(trajectory: List[int], S: int, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    # k: number of skips
    trajectory = np.array(trajectory)  # Convert to numpy array
    H = len(trajectory)
    
    # Vectorized computation using advanced indexing
    h_indices = np.arange(0, (H - 1) // k) * k
    s_h_values = trajectory[h_indices]
    s_next_values = trajectory[h_indices + k]
    
    # Use bincount for efficient counting - much faster than loops
    N_hat_ = np.bincount(s_h_values, minlength=S)
    
    # For transition matrix, use advanced indexing with bincount
    # Create linear indices for 2D array
    linear_indices = s_h_values * S + s_next_values
    N_hat_flat = np.bincount(linear_indices, minlength=S*S)
    N_hat = N_hat_flat.reshape(S, S)
        
    return N_hat, N_hat_

def L_embedding(trajectories: List[List[int]], T: int, H: int, S: int) -> np.ndarray:
    W_hat = np.zeros((T, S**2))
    
    # Batch process all trajectories for better cache locality
    for t, trajectory in enumerate(trajectories):
        N_hat, N_hat_ = empirical_transition_matrix(trajectory, S)
        
        # Vectorized computation of L_hat_t
        # Avoid division by zero
        sqrt_H_N = np.sqrt(H * N_hat_)
        sqrt_H_N[sqrt_H_N == 0] = 1  # Avoid division by zero
        L_hat_t = np.diag(1 / sqrt_H_N) @ N_hat
        W_hat[t] = L_hat_t.flatten('C')
    
    return W_hat

def InitialSpectral(trajectories: List[List[int]], T: int, H: int, S: int, gamma_ps: float, delta: float, verbose: bool = False) -> Dict[int, int]:
    f_hat = {}

    # SVD with thresholding
    W_hat = L_embedding(trajectories, T, H, S)
    # sigma_threshold = 8 * np.sqrt(T * S * np.log(T*H/delta) / (H * gamma_ps))
    sigma_threshold = SIGMA_THRESHOLD_FACTOR * np.sqrt(T * S * np.log(T*H/delta) / (H * gamma_ps))
    U, Sigma, _ = LA.svd(W_hat)
    if verbose:
        print(Sigma)
    # plt.plot(S)
    # plt.show()
    if verbose:
        print(sigma_threshold)
    R_hat = sum(Sigma >= sigma_threshold)
    X_hat = U[:, :R_hat] @ np.diag(Sigma[:R_hat])
    if verbose:
        print(R_hat)

    # Density-based clustering
    # Pre-compute distance matrix for efficiency
    if verbose:
        print("Computing distance matrix...")
    distances = np.linalg.norm(X_hat[:, np.newaxis, :] - X_hat[np.newaxis, :, :], axis=2)
    within_threshold = distances <= sigma_threshold
    
    # Convert to sets for compatibility with existing algorithm
    Q_dict = {t: set(np.where(within_threshold[t])[0]) for t in range(T)}
    
    center_idxs = {}
    S_union = set()
    S_dict = {0: set()}
    k, rho = 1, T
    
    # Cache the threshold for repeated use
    rho_threshold = RHO_THRESHOLD_FACTOR * R_hat*T / np.log(T*H / delta)
    
    while rho >= rho_threshold:
        S_union = S_union.union(S_dict[k-1])
        
        # Vectorized computation of set differences for better performance
        remaining_counts = np.array([len(Q_dict[t] - S_union) for t in range(T)])
        t_k_star = np.argmax(remaining_counts)
        
        center_idxs[k] = t_k_star
        S_star = Q_dict[t_k_star] - S_union
        S_dict[k] = S_star
        k += 1
        rho = len(S_star)
    K_hat = int(k - 1)

    for k in range(K_hat):
        for t in S_dict[k+1]:
            f_hat[t] = k

    # K-means clustering for the remaining
    remaining_t = set(range(T)) - S_union
    if remaining_t:
        # Use pre-computed distances for efficiency
        remaining_arr = np.array(list(remaining_t))
        center_arr = np.array([center_idxs[k+1] for k in range(K_hat)])
        remaining_distances = distances[np.ix_(remaining_arr, center_arr)]
        assignments = np.argmin(remaining_distances, axis=1)
        
        for i, t in enumerate(remaining_arr):
            f_hat[t] = assignments[i]

    if verbose:
        print(S_dict)
        print(f_hat)
    return f_hat



def clustered_transition_matrix(trajectories, indices, S, verbose: bool = False):
    if not indices:
        # Return uniform distribution if no trajectories
        return np.ones((S, S)) / S
    
    # Vectorized accumulation of transition matrices
    N_hat, N_hat_ = np.zeros((S, S)), np.zeros(S)
    
    # Process all trajectories in the cluster
    for t in indices:
        N_hat_t, N_hat_t_ = empirical_transition_matrix(trajectories[t], S)
        N_hat += N_hat_t
        N_hat_ += N_hat_t_
    
    # Handle zero counts - use uniform distribution for unobserved states
    zero_mask = N_hat_ == 0
    N_hat_[zero_mask] = 1  # Avoid division by zero
    N_hat[zero_mask] = np.ones(S) / S  # Uniform distribution
    
    return N_hat / N_hat_[:, np.newaxis]
    
def log_likelihood(trajectory, P, verbose: bool = False):
    trajectory = np.array(trajectory)
    if len(trajectory) < 2:
        return 0.0
    if verbose:
        print(trajectory)
        print(P)
    
    # Vectorized computation using advanced indexing
    current_states = trajectory[:-1]
    next_states = trajectory[1:]
    
    # Get transition probabilities in one operation
    transition_probs = P[current_states, next_states]
    
    # Avoid log(0) by adding small epsilon
    transition_probs = np.maximum(transition_probs, LOG_EPSILON)
    
    return np.sum(np.log(transition_probs))

def LikelihoodRefinement(trajectories, f_hat_1, T, S, max_iter=1, verbose: bool = False):
    f_hat = f_hat_1.copy()
    K_hat = len(set(f_hat_1.values()))
    
    for l in range(max_iter):
        # Compute transition matrices for each cluster
        P_dict = {k: clustered_transition_matrix(trajectories, [t for t in range(T) if f_hat[t] == k], S, verbose) for k in range(K_hat)}
        
        # Vectorized likelihood computation for all trajectories and clusters
        likelihoods = np.zeros((T, K_hat))
        for t in range(T):
            trajectory = np.array(trajectories[t])
            if len(trajectory) < 2:
                continue
                
            current_states = trajectory[:-1]
            next_states = trajectory[1:]
            
            for k in range(K_hat):
                # Vectorized computation for cluster k
                transition_probs = P_dict[k][current_states, next_states]
                transition_probs = np.maximum(transition_probs, LOG_EPSILON)
                likelihoods[t, k] = np.sum(np.log(transition_probs))
        if verbose:
            print(likelihoods)
        # Assign each trajectory to the cluster with highest likelihood
        f_hat = {t: int(np.argmax(likelihoods[t])) for t in range(T)}
    if verbose:
        print(f_hat)
    return f_hat