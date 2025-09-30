"""
Stage 3: EM Algorithm for Mixture of Markov Chains (No Actions)

This module implements the EM algorithm stage of the three-stage algorithm
for learning mixtures of Markov chains, adapted from the mdpmix approach but
simplified for the no-action scenario.

The EM algorithm refines the initial clustering results by iteratively updating
transition matrices and cluster assignments to maximize the likelihood of the data.
"""

import numpy as np
from numba import jit, njit, prange
from tqdm import tqdm
import warnings
from typing import List, Tuple, Dict, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@njit(parallel=False, cache=True)
def compute_transition_counts(trajectories, cluster_labels, K, n_states):
    """
    Compute transition counts for each cluster.
    
    Args:
        trajectories: List of trajectories (as numpy arrays)
        cluster_labels: Cluster assignments for each trajectory
        K: Number of clusters
        n_states: Number of states
        
    Returns:
        transition_counts: Array of shape (K, n_states, n_states)
        state_counts: Array of shape (K, n_states)
    """
    transition_counts = np.zeros((K, n_states, n_states))
    state_counts = np.zeros((K, n_states))
    
    for traj_idx in range(len(trajectories)):
        trajectory = trajectories[traj_idx]
        cluster = cluster_labels[traj_idx]
        
        for t in range(len(trajectory) - 1):
            current_state = trajectory[t]
            next_state = trajectory[t + 1]
            
            transition_counts[cluster, current_state, next_state] += 1
            state_counts[cluster, current_state] += 1
    
    return transition_counts, state_counts

@njit(parallel=False, cache=True)
def compute_transition_counts_soft(trajectories, soft_assignments, K, n_states):
    """
    Compute transition counts for each cluster using soft assignments.
    
    Args:
        trajectories: List of trajectories (as numpy arrays)
        soft_assignments: Soft cluster assignments of shape (K, n_trajectories)
        K: Number of clusters
        n_states: Number of states
        
    Returns:
        transition_counts: Array of shape (K, n_states, n_states)
        state_counts: Array of shape (K, n_states)
    """
    transition_counts = np.zeros((K, n_states, n_states))
    state_counts = np.zeros((K, n_states))
    
    for traj_idx in range(len(trajectories)):
        trajectory = trajectories[traj_idx]
        
        for t in range(len(trajectory) - 1):
            current_state = trajectory[t]
            next_state = trajectory[t + 1]
            
            for k in range(K):
                weight = soft_assignments[k, traj_idx]
                transition_counts[k, current_state, next_state] += weight
                state_counts[k, current_state] += weight
    
    return transition_counts, state_counts

def estimate_transition_matrices(transition_counts, state_counts, smoothing=0.0):
    """
    Estimate transition matrices from counts with smoothing.
    
    Args:
        transition_counts: Array of shape (K, n_states, n_states)
        state_counts: Array of shape (K, n_states)
        smoothing: Smoothing parameter to avoid zero probabilities
        
    Returns:
        transition_matrices: Array of shape (K, n_states, n_states)
    """
    K, n_states, _ = transition_counts.shape
    transition_matrices = np.zeros((K, n_states, n_states))
    
    for k in range(K):
        for s in range(n_states):
            total_count = state_counts[k, s]
            
            if total_count > 0:
                # Normalize by state counts
                transition_matrices[k, s, :] = transition_counts[k, s, :] / total_count
            else:
                # Use uniform distribution if no data
                transition_matrices[k, s, :] = 1.0 / n_states
            
            # Apply optional smoothing then renormalize
            if smoothing > 0:
                transition_matrices[k, s, :] += smoothing
                transition_matrices[k, s, :] /= np.sum(transition_matrices[k, s, :])
    
    return transition_matrices

def compute_trajectory_likelihood(trajectory, transition_matrix, start_prob=None):
    """
    Compute log-likelihood of a trajectory under a transition matrix.
    
    Args:
        trajectory: List or array of states
        transition_matrix: Transition matrix of shape (n_states, n_states)
        start_prob: Starting state probabilities (optional)
        
    Returns:
        log_likelihood: Log-likelihood of the trajectory
    """
    trajectory = np.array(trajectory)
    log_likelihood = 0.0
    
    # Add starting state probability if provided
    if start_prob is not None:
        log_likelihood += np.log(start_prob[trajectory[0]] + 1e-10)
    
    # Add transition probabilities
    for t in range(len(trajectory) - 1):
        current_state = trajectory[t]
        next_state = trajectory[t + 1]
        prob = transition_matrix[current_state, next_state]
        log_likelihood += np.log(prob + 1e-10)
    
    return log_likelihood

def compute_soft_assignments(trajectories, transition_matrices, cluster_priors, 
                           start_probs=None, verbose=False):
    """
    Compute soft cluster assignments (E-step).
    
    Args:
        trajectories: List of trajectories
        transition_matrices: Array of shape (K, n_states, n_states)
        cluster_priors: Prior probabilities for each cluster
        start_probs: Starting state probabilities for each cluster
        verbose: Print progress information
        
    Returns:
        soft_assignments: Array of shape (K, n_trajectories)
    """
    K = len(transition_matrices)
    n_trajectories = len(trajectories)
    
    # Compute log-likelihoods for each trajectory under each cluster
    log_likelihoods = np.zeros((K, n_trajectories))
    
    for k in range(K):
        for traj_idx, trajectory in enumerate(trajectories):
            start_prob = start_probs[k] if start_probs is not None else None
            log_likelihoods[k, traj_idx] = compute_trajectory_likelihood(
                trajectory, transition_matrices[k], start_prob
            )
    
    # Add log priors
    log_likelihoods += np.log(cluster_priors + 1e-10)[:, np.newaxis]
    
    # Compute soft assignments using log-sum-exp trick
    max_log_likelihood = np.max(log_likelihoods, axis=0)
    exp_likelihoods = np.exp(log_likelihoods - max_log_likelihood)
    soft_assignments = exp_likelihoods / np.sum(exp_likelihoods, axis=0)
    
    if verbose:
        print(f"Soft assignments computed. Shape: {soft_assignments.shape}")
        print(f"Assignment entropy: {np.mean(-np.sum(soft_assignments * np.log(soft_assignments + 1e-10), axis=0)):.4f}")
    
    return soft_assignments

def update_cluster_priors(soft_assignments):
    """
    Update cluster priors based on soft assignments (M-step).
    
    Args:
        soft_assignments: Array of shape (K, n_trajectories)
        
    Returns:
        cluster_priors: Updated prior probabilities
    """
    return np.mean(soft_assignments, axis=1)

def estimate_starting_probabilities(trajectories, soft_assignments, n_states):
    """
    Estimate starting state probabilities for each cluster.
    
    Args:
        trajectories: List of trajectories
        soft_assignments: Soft cluster assignments
        n_states: Number of states
        
    Returns:
        start_probs: Array of shape (K, n_states)
    """
    K = soft_assignments.shape[0]
    start_counts = np.zeros((K, n_states))
    cluster_counts = np.sum(soft_assignments, axis=1)
    
    for traj_idx, trajectory in enumerate(trajectories):
        first_state = trajectory[0]
        for k in range(K):
            start_counts[k, first_state] += soft_assignments[k, traj_idx]
    
    # Normalize
    start_probs = np.zeros((K, n_states))
    for k in range(K):
        if cluster_counts[k] > 0:
            start_probs[k, :] = start_counts[k, :] / cluster_counts[k]
        else:
            start_probs[k, :] = 1.0 / n_states
    
    return start_probs

def em_algorithm(trajectories, initial_cluster_labels, K, n_states, 
                max_iterations=100, min_iterations=10, tolerance=1e-3,
                hard_assignments=True, verbose=False):
    """
    Main EM algorithm for mixture of Markov chains.
    
    Args:
        trajectories: List of trajectories
        initial_cluster_labels: Initial cluster assignments
        K: Number of clusters
        n_states: Number of states
        max_iterations: Maximum number of iterations
        min_iterations: Minimum number of iterations
        tolerance: Convergence tolerance
        hard_assignments: Whether to use hard assignments (True) or soft (False)
        verbose: Print progress information
        
    Returns:
        results: Dictionary containing:
            - transition_matrices: Final transition matrices
            - cluster_labels: Final cluster assignments
            - cluster_priors: Final cluster priors
            - start_probs: Starting state probabilities
            - log_likelihoods: Log-likelihood per iteration
            - converged: Whether algorithm converged
    """
    if verbose:
        print(f"\nStage 3: EM Algorithm")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Number of clusters: {K}")
        print(f"Number of states: {n_states}")
        print(f"Using {'hard' if hard_assignments else 'soft'} assignments")
    
    # Convert trajectories to numpy arrays for efficiency
    trajectories = [np.array(traj) for traj in trajectories]
    
    # Initialize
    cluster_labels = np.array(initial_cluster_labels)
    log_likelihoods = []
    
    for iteration in range(max_iterations):
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}...")
        
        # E-step: Compute assignments
        if hard_assignments:
            # Hard assignments: use current cluster labels
            transition_counts, state_counts = compute_transition_counts(
                trajectories, cluster_labels, K, n_states
            )
            # Match mdpmix prior smoothing (Laplace-like +1 across clusters)
            cluster_counts = np.bincount(cluster_labels, minlength=K)
            cluster_priors = (cluster_counts + 1) / (len(trajectories) + K)
        else:
            # Soft assignments: compute soft assignments
            if iteration == 0:
                # Initialize soft assignments from hard assignments
                soft_assignments = np.zeros((K, len(trajectories)))
                for i, label in enumerate(cluster_labels):
                    soft_assignments[label, i] = 1.0
            else:
                # Compute soft assignments from current models
                soft_assignments = compute_soft_assignments(
                    trajectories, transition_matrices, cluster_priors, start_probs
                )
            
            transition_counts, state_counts = compute_transition_counts_soft(
                trajectories, soft_assignments, K, n_states
            )
            cluster_priors = update_cluster_priors(soft_assignments)
        
        # M-step: Update transition matrices
        transition_matrices = estimate_transition_matrices(transition_counts, state_counts)
        
        # Estimate starting probabilities
        if hard_assignments:
            start_counts = np.zeros((K, n_states))
            for i, trajectory in enumerate(trajectories):
                first_state = trajectory[0]
                cluster = cluster_labels[i]
                start_counts[cluster, first_state] += 1
            
            start_probs = np.zeros((K, n_states))
            cluster_counts = np.bincount(cluster_labels, minlength=K)
            for k in range(K):
                if cluster_counts[k] > 0:
                    start_probs[k, :] = start_counts[k, :] / cluster_counts[k]
                else:
                    start_probs[k, :] = 1.0 / n_states
        else:
            start_probs = estimate_starting_probabilities(trajectories, soft_assignments, n_states)
        
        # Compute log-likelihood
        total_log_likelihood = 0.0
        for i, trajectory in enumerate(trajectories):
            if hard_assignments:
                cluster = cluster_labels[i]
                ll = compute_trajectory_likelihood(trajectory, transition_matrices[cluster], start_probs[cluster])
                total_log_likelihood += ll
            else:
                for k in range(K):
                    weight = soft_assignments[k, i]
                    ll = compute_trajectory_likelihood(trajectory, transition_matrices[k], start_probs[k])
                    total_log_likelihood += weight * ll
        
        log_likelihoods.append(total_log_likelihood)
        
        # Update hard assignments if using hard EM
        if hard_assignments:
            new_cluster_labels = np.zeros(len(trajectories), dtype=int)
            for i, trajectory in enumerate(trajectories):
                best_cluster = 0
                best_likelihood = float('-inf')
                
                for k in range(K):
                    ll = compute_trajectory_likelihood(trajectory, transition_matrices[k], start_probs[k])
                    # tiny jitter to break ties as in mdpmix
                    ll += np.random.uniform(high=1e-7)
                    ll += np.log(cluster_priors[k] + 1e-10)
                    
                    if ll > best_likelihood:
                        best_likelihood = ll
                        best_cluster = k
                
                new_cluster_labels[i] = best_cluster
            
            # Check convergence
            if iteration >= min_iterations:
                if np.array_equal(cluster_labels, new_cluster_labels):
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
            
            cluster_labels = new_cluster_labels
        
        else:
            # For soft EM, check parameter convergence similar to mdpmix (loglik as proxy kept)
            if iteration >= min_iterations and len(log_likelihoods) >= 2:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
    
    # Convert soft assignments to hard assignments for final result
    if not hard_assignments:
        cluster_labels = np.argmax(soft_assignments, axis=0)
    
    results = {
        'transition_matrices': transition_matrices,
        'cluster_labels': cluster_labels,
        'cluster_priors': cluster_priors,
        'start_probs': start_probs,
        'log_likelihoods': log_likelihoods,
        'converged': iteration < max_iterations - 1,
        'iterations': iteration + 1
    }
    
    if verbose:
        print(f"EM completed after {results['iterations']} iterations")
        print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
        print(f"Cluster sizes: {np.bincount(cluster_labels)}")
    
    return results

def em_stage(trajectories, cluster_labels, K, n_states=None, 
            max_iterations=100, hard_assignments=True, verbose=False):
    """
    Main function for Stage 3: EM Algorithm.
    
    This is the entry point for the third stage of the three-stage algorithm.
    
    Args:
        trajectories: List of trajectories, each as list of states
        cluster_labels: Initial cluster assignments from Stage 2
        K: Number of mixture components
        n_states: Number of states (inferred if None)
        max_iterations: Maximum number of EM iterations
        hard_assignments: Whether to use hard assignments
        verbose: Print progress information
        
    Returns:
        results: Dictionary with EM results
    """
    if verbose:
        print(f"\nStage 3: EM Algorithm")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Number of clusters: {K}")
    
    # Infer number of states if not provided
    if n_states is None:
        n_states = max(max(traj) for traj in trajectories) + 1
    
    if verbose:
        print(f"Number of states: {n_states}")
    
    # Run EM algorithm
    results = em_algorithm(
        trajectories, cluster_labels, K, n_states,
        max_iterations=max_iterations, hard_assignments=hard_assignments,
        verbose=verbose
    )
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test with simple synthetic data
    np.random.seed(42)
    
    # Create simple synthetic trajectories (same as previous stages)
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
    true_labels = []
    for i in range(n_trajectories):
        if i < n_trajectories // 2:
            P = P1
            true_labels.append(0)
        else:
            P = P2
            true_labels.append(1)
            
        trajectory = [np.random.choice(n_states)]
        for t in range(trajectory_length - 1):
            next_state = np.random.choice(n_states, p=P[trajectory[-1], :])
            trajectory.append(next_state)
        trajectories.append(trajectory)
    
    # Create some initial cluster labels (simulate Stage 2 output)
    initial_labels = np.random.randint(0, 2, n_trajectories)
    
    # Test Stage 3
    print("Testing Stage 3: EM Algorithm...")
    results = em_stage(
        trajectories, initial_labels, K=2, 
        max_iterations=50, hard_assignments=True, verbose=True
    )
    
    print(f"\nResults:")
    print(f"Final cluster labels: {results['cluster_labels']}")
    print(f"True labels: {true_labels}")
    print(f"Cluster priors: {results['cluster_priors']}")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    
    # Compute accuracy
    from scipy.optimize import linear_sum_assignment
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(results['cluster_labels'])):
        confusion_matrix[results['cluster_labels'][i], true_labels[i]] += 1
    
    costs = 2 * 2 - confusion_matrix
    row_ind, col_ind = linear_sum_assignment(costs)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    
    print(f"Clustering accuracy: {accuracy:.4f}")

