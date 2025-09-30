"""
Stage 1: Subspace Estimation for Mixture of Markov Chains (No Actions)

This module implements the subspace estimation stage of the three-stage algorithm
for learning mixtures of Markov chains, adapted from the mdpmix approach but
simplified for the no-action scenario.

The key idea is that parameters for K models in a mixture should lie in the 
K-dimensional subspace spanned by them. This stage estimates this subspace by
aggregating across trajectories using trajectory partitioning.
"""

import numpy as np
from numba import jit, njit, prange
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@njit(parallel=True, cache=True)
def get_empirical_transition_matrix(onehot_states, onehot_next_states, simple=False):
    """
    Compute empirical transition probabilities from state-action-next_state data.
    
    For no-action case, we only use state-next_state transitions.
    
    Args:
        onehot_states: Array of shape (m, t, s) - one-hot encoded current states
        onehot_next_states: Array of shape (m, t, s) - one-hot encoded next states  
        simple: If True, normalize by trajectory length instead of counts
        
    Returns:
        h: Array of shape (m, s, s') - empirical transition probabilities
        N_ms: Array of shape (m, s) - state counts
    """
    m, t, s = onehot_states.shape
    h = np.zeros((m, s, s))
    N_ms = np.zeros((m, s))
    
    for traj_idx in prange(m):
        for state_idx in range(s):
            for next_state_idx in range(s):
                for time_idx in range(t):
                    h[traj_idx, state_idx, next_state_idx] += (
                        onehot_states[traj_idx, time_idx, state_idx] * 
                        onehot_next_states[traj_idx, time_idx, next_state_idx]
                    )
                    N_ms[traj_idx, state_idx] += onehot_states[traj_idx, time_idx, state_idx]
    
    # Normalize transition probabilities
    if not simple:
        for traj_idx in range(m):
            for state_idx in range(s):
                if N_ms[traj_idx, state_idx] != 0:
                    h[traj_idx, state_idx, :] /= N_ms[traj_idx, state_idx]
                else:
                    # Match mdpmix behavior: leave zeros when no counts
                    h[traj_idx, state_idx, :] = 0
    else:
        h /= t
    
    return h, N_ms

def get_subspace_projections(trajectories, omega_one, omega_two, K, weights=True, 
                           verbose=False, device='/CPU:0'):
    """
    Compute projections of transition probabilities to rank K subspaces.
    
    This is the core subspace estimation function adapted from mdpmix but for 
    the no-action case (Markov chains instead of MDPs).
    
    Args:
        trajectories: List of trajectories, each as list of states
        omega_one: Indices for first partition (Ω₁)
        omega_two: Indices for second partition (Ω₂)  
        K: Number of mixture components
        weights: Whether to use trajectory weights
        verbose: Print progress information
        device: Device for computation (for future GPU support)
        
    Returns:
        eigvals: Eigenvalues of shape (s, K) 
        eigvecs: Eigenvectors of shape (s, s, K)
    """
    if verbose:
        print(f"Computing subspace projections for K={K} components...")
        print(f"Using partitions: Ω₁={len(omega_one)} timesteps, Ω₂={len(omega_two)} timesteps")
    
    # Convert trajectories to one-hot encoding
    max_state = max(max(traj) for traj in trajectories)
    n_states = max_state + 1
    n_trajectories = len(trajectories)
    
    # Create one-hot encoded states for both partitions
    onehot_states_1 = np.zeros((n_trajectories, len(omega_one), n_states))
    onehot_next_states_1 = np.zeros((n_trajectories, len(omega_one), n_states))
    
    onehot_states_2 = np.zeros((n_trajectories, len(omega_two), n_states))
    onehot_next_states_2 = np.zeros((n_trajectories, len(omega_two), n_states))
    
    for traj_idx, trajectory in enumerate(trajectories):
        for i, t in enumerate(omega_one):
            if t < len(trajectory) - 1:  # Ensure we have next state
                onehot_states_1[traj_idx, i, trajectory[t]] = 1
                onehot_next_states_1[traj_idx, i, trajectory[t+1]] = 1
        
        for i, t in enumerate(omega_two):
            if t < len(trajectory) - 1:  # Ensure we have next state
                onehot_states_2[traj_idx, i, trajectory[t]] = 1
                onehot_next_states_2[traj_idx, i, trajectory[t+1]] = 1
    
    # Compute empirical transition matrices for both partitions
    h1, _ = get_empirical_transition_matrix(onehot_states_1, onehot_next_states_1)
    h2, _ = get_empirical_transition_matrix(onehot_states_2, onehot_next_states_2)
    
    if verbose:
        print(f"Computed transition matrices of shape {h1.shape}")
    
    # Compute trajectory weights if requested
    if weights:
        # Compute state visit counts for weighting
        traj_weights = (onehot_states_1.sum(axis=1) * onehot_states_2.sum(axis=1)).sum(axis=0)
        inv_weights = 1.0 / traj_weights
        inv_weights[np.isinf(inv_weights)] = 0
    else:
        inv_weights = np.ones(n_states)
    
    # Compute per-state H matrices for eigendecomposition, matching mdpmix logic
    # For each state s: H_s = mean_m( (h1[m, s, :] ⊗ h2[m, s, :]) ) with weighting
    H_states = np.zeros((n_states, n_states, n_states))  # s, s', s''
    for traj_idx in tqdm(range(n_trajectories), desc="Computing H matrices per state", disable=not verbose):
        for state_idx in range(n_states):
            outer_prod = np.outer(h1[traj_idx, state_idx, :], h2[traj_idx, state_idx, :])
            H_states[state_idx, :, :] += outer_prod * inv_weights[state_idx]
    # Match mdpmix small-data behavior: do not divide by number of trajectories

    # Symmetrize and eigendecompose per state
    eigvals = np.zeros((n_states, n_states))
    eigvecs = np.zeros((n_states, n_states, n_states))
    for state_idx in range(n_states):
        H_state = H_states[state_idx, :, :]
        H_state = H_state + H_state.T
        vals, vecs = np.linalg.eigh(H_state)
        # Match mdpmix: eigenvalues returned ascending; take last K later without re-sorting
        eigvals[state_idx, :] = vals
        eigvecs[state_idx, :, :] = vecs
    
    # Return top K eigenvalues and eigenvectors
    return eigvals[:, -K:], eigvecs[:, :, -K:]

def get_occupancy_subspace(trajectories, omega_one, omega_two, K, verbose=False):
    """
    Compute projections of occupancy measures to rank K subspaces.
    
    This is a simpler version that uses state occupancy rather than transitions.
    
    Args:
        trajectories: List of trajectories
        omega_one: Indices for first partition
        omega_two: Indices for second partition
        K: Number of mixture components
        verbose: Print progress information
        
    Returns:
        eigvals: Eigenvalues of shape (K,)
        eigvecs: Eigenvectors of shape (n_states, K)
    """
    if verbose:
        print(f"Computing occupancy subspace for K={K} components...")
    
    # Convert trajectories to one-hot encoding for occupancy
    max_state = max(max(traj) for traj in trajectories)
    n_states = max_state + 1
    n_trajectories = len(trajectories)
    
    # Compute occupancy measures for both partitions
    occupancy_1 = np.zeros((n_trajectories, n_states))
    occupancy_2 = np.zeros((n_trajectories, n_states))
    
    for traj_idx, trajectory in enumerate(trajectories):
        for t in omega_one:
            if t < len(trajectory):
                occupancy_1[traj_idx, trajectory[t]] += 1
        
        for t in omega_two:
            if t < len(trajectory):
                occupancy_2[traj_idx, trajectory[t]] += 1
        
        # Normalize by partition length
        occupancy_1[traj_idx, :] /= len(omega_one)
        occupancy_2[traj_idx, :] /= len(omega_two)
    
    # Compute the matrix K for eigendecomposition
    # K = mean(occupancy_1 ⊗ occupancy_2)
    K_matrix = np.zeros((n_states, n_states))
    for traj_idx in range(n_trajectories):
        outer_prod = np.outer(occupancy_1[traj_idx, :], occupancy_2[traj_idx, :])
        K_matrix += outer_prod
    
    K_matrix /= n_trajectories
    
    # Make symmetric
    K_matrix = K_matrix + K_matrix.T
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(K_matrix)
    
    # Sort by eigenvalue magnitude (descending)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    
    # Return top K eigenvalues and eigenvectors
    return eigvals[:K], eigvecs[:, :K]

def partition_trajectories(trajectory_length, num_partitions=2, overlap=0.1):
    """
    Create trajectory partitions for subspace estimation.
    
    Args:
        trajectory_length: Length of trajectories
        num_partitions: Number of partitions (typically 2)
        overlap: Fraction of overlap between partitions
        
    Returns:
        partitions: List of partition indices
    """
    partitions = []
    
    if num_partitions == 2:
        # Standard two-partition approach from mdpmix
        partition_size = trajectory_length // 4  # Use quarter of trajectory for each partition
        
        # First partition: middle quarter
        omega_one = list(range(partition_size, 2 * partition_size))
        
        # Second partition: last quarter  
        omega_two = list(range(3 * partition_size, 4 * partition_size))
        
        partitions = [omega_one, omega_two]
    
    else:
        # General case: divide trajectory into non-overlapping segments
        segment_size = trajectory_length // num_partitions
        for i in range(num_partitions):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, trajectory_length)
            partitions.append(list(range(start_idx, end_idx)))
    
    return partitions

def subspace_estimation_stage(trajectories, K, use_transitions=True, verbose=False):
    """
    Main function for Stage 1: Subspace Estimation.
    
    This is the entry point for the first stage of the three-stage algorithm.
    
    Args:
        trajectories: List of trajectories, each as list of states
        K: Number of mixture components (can be None for adaptive estimation)
        use_transitions: Whether to use transition-based or occupancy-based estimation
        verbose: Print progress information
        
    Returns:
        eigvals: Eigenvalues 
        eigvecs: Eigenvectors
        partitions: Trajectory partitions used
    """
    if not trajectories:
        raise ValueError("No trajectories provided")
    
    # Check trajectory lengths
    trajectory_lengths = [len(traj) for traj in trajectories]
    min_length = min(trajectory_lengths)
    
    if min_length < 4:
        raise ValueError(f"Trajectories too short (min length: {min_length}, need at least 4)")
    
    # Use minimum trajectory length for partitioning
    trajectory_length = min_length
    
    if verbose:
        print(f"Stage 1: Subspace Estimation")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Trajectory length: {trajectory_length}")
        print(f"Number of components: {K}")
    
    # Create trajectory partitions
    partitions = partition_trajectories(trajectory_length)
    omega_one, omega_two = partitions
    
    if verbose:
        print(f"Using partitions: Ω₁={len(omega_one)}, Ω₂={len(omega_two)}")
    
    # Estimate subspace
    if use_transitions:
        eigvals, eigvecs = get_subspace_projections(
            trajectories, omega_one, omega_two, K, verbose=verbose
        )
    else:
        eigvals, eigvecs = get_occupancy_subspace(
            trajectories, omega_one, omega_two, K, verbose=verbose
        )
    
    if verbose:
        print(f"Subspace estimation completed. Eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
    
    return eigvals, eigvecs, partitions

# Example usage and testing
if __name__ == "__main__":
    # Test with simple synthetic data
    np.random.seed(42)
    
    # Create simple synthetic trajectories
    n_trajectories = 50
    trajectory_length = 100
    n_states = 5
    
    # Create two different transition matrices
    P1 = np.array([
        [0.7, 0.2, 0.1, 0.0, 0.0],
        [0.1, 0.7, 0.2, 0.0, 0.0], 
        [0.0, 0.1, 0.7, 0.2, 0.0],
        [0.0, 0.0, 0.1, 0.7, 0.2],
        [0.2, 0.0, 0.0, 0.1, 0.7]
    ])
    
    P2 = np.array([
        [0.2, 0.3, 0.3, 0.2, 0.0],
        [0.3, 0.2, 0.3, 0.2, 0.0],
        [0.2, 0.3, 0.2, 0.3, 0.0], 
        [0.0, 0.2, 0.3, 0.2, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2]
    ])
    
    # Generate trajectories
    trajectories = []
    for i in range(n_trajectories):
        if i < n_trajectories // 2:
            P = P1
        else:
            P = P2
            
        trajectory = [np.random.choice(n_states)]
        for t in range(trajectory_length - 1):
            next_state = np.random.choice(n_states, p=P[trajectory[-1], :])
            trajectory.append(next_state)
        trajectories.append(trajectory)
    
    # Test subspace estimation
    print("Testing Stage 1: Subspace Estimation")
    eigvals, eigvecs, partitions = subspace_estimation_stage(
        trajectories, K=2, use_transitions=True, verbose=True
    )
    
    print(f"\nResults:")
    print(f"Eigenvalues shape: {eigvals.shape}")
    print(f"Eigenvectors shape: {eigvecs.shape}")
    print(f"Top eigenvalues: {eigvals[0, :]}")
