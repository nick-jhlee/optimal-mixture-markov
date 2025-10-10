import numpy as np
import numpy.linalg as LA
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from utils import log_likelihood, empirical_visitation_matrix, clustered_transition_matrix

# Constants for optimization
SIGMA_THRESHOLD_FACTOR = 1e-4  # Factor for sigma threshold calculation
RHO_THRESHOLD_FACTOR = 1e-1    # Factor for rho threshold calculation


## Initial spectral clustering
def L_embedding(trajectories: List[List[int]], T: int, H: int, S: int) -> np.ndarray:
    W_hat = np.zeros((T, S**2))
    
    # Batch process all trajectories for better cache locality
    for t, trajectory in enumerate(trajectories):
        N_hat, N_hat_ = empirical_visitation_matrix(trajectory, S)
        
        # Vectorized computation of L_hat_t
        # Avoid division by zero
        sqrt_H_N = np.sqrt(H * N_hat_)
        sqrt_H_N[sqrt_H_N == 0] = 1  # Avoid division by zero
        L_hat_t = np.diag(1 / sqrt_H_N) @ N_hat
        W_hat[t] = L_hat_t.flatten('C')
    
    return W_hat

def InitialSpectral(trajectories: List[List[int]], T: int, H: int, S: int, gamma_ps: float, delta: float, K: int = None, c1: Optional[float] = None, c2: Optional[float] = None, verbose: bool = False) -> Dict[int, int]:
    f_hat = {}

    # SVD
    W_hat = L_embedding(trajectories, T, H, S)
    U, Sigma, _ = LA.svd(W_hat)
    if verbose:
        print(Sigma)

    # K is not known => adaptive clustering procedure
    if K is None:
        # Use provided c1, c2 if given; otherwise fall back to module defaults
        sigma_factor = SIGMA_THRESHOLD_FACTOR if c1 is None else float(c1)
        rho_factor = RHO_THRESHOLD_FACTOR if c2 is None else float(c2)
        sigma_threshold = np.sqrt(sigma_factor * T * S * np.log(T*H/delta) / (H * gamma_ps))
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
        rho_threshold = rho_factor * R_hat*T / np.log(T*H / delta)
        
        while rho > rho_threshold:
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

        # If no centers are discovered, fall back to a single cluster
        if K_hat <= 0:
            for t in range(T):
                f_hat[t] = 0
            if verbose:
                print("K_hat == 0; assigning all trajectories to a single cluster.")
            return f_hat

        for k in range(K_hat):
            for t in S_dict[k+1]:
                f_hat[t] = k

        # K-means clustering for the remaining
        remaining_t = set(range(T)) - S_union
        if remaining_t:
            # Use pre-computed distances for efficiency; ensure integer index dtype
            remaining_arr = np.array(list(remaining_t), dtype=int)
            center_arr = np.array([center_idxs[k+1] for k in range(K_hat)], dtype=int)
            remaining_distances = distances[np.ix_(remaining_arr, center_arr)]
            assignments = np.argmin(remaining_distances, axis=1)
            
            for i, t in enumerate(remaining_arr):
                f_hat[t] = assignments[i]

        if verbose:
            print(S_dict)
            print(f_hat)
        return f_hat
    # K is known => the usual spectral clsutering
    else:
        X_hat = U[:, :K] @ np.diag(Sigma[:K])
        # K-means clustering for the rows
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X_hat)
        f_hat = {t: kmeans.labels_[t] for t in range(T)}
        return f_hat


## Likelihood-based refinement
def OracleLikelihoodRefinement(trajectories, env):
    Ps = env.Ps
    mus = env.mus

    f_oracle = {}
    for t, trajectory in enumerate(trajectories):
        likelihoods_t = [log_likelihood(trajectory, P, mu) for P, mu in zip(Ps, mus)]
        f_oracle[t] = np.argmax(likelihoods_t)
    return f_oracle

def LikelihoodRefinement(trajectories, f_hat_1, T, S, max_iter: int = 1, history: bool = False):
    f_hat = f_hat_1.copy()
    K_hat = len(set(f_hat_1.values()))
    H = len(trajectories[0])

    f_hats = []
    logliks = []  # total log-likelihood per iteration (after reassignment)
    for _ in range(max_iter):
        # Compute transition matrices for each cluster
        Ps_hat = [clustered_transition_matrix(trajectories, [t for t in range(T) if f_hat[t] == k], S) for k in range(K_hat)]

        total_likelihood = 0.0
        for t, trajectory in enumerate(trajectories):
            likelihoods_t = [log_likelihood(trajectory, P) for P in Ps_hat]
            f_hat[t] = np.argmax(likelihoods_t)
            total_likelihood += float(np.max(likelihoods_t) / (T * H))
        
        if history:
            # Store an immutable snapshot for this iteration
            f_hats.append(f_hat.copy())
            logliks.append(total_likelihood)
    
    if history:
        return f_hats, np.array(logliks, dtype=float)
    else:
        return f_hat
