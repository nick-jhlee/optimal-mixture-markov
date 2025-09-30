"""
Stage 2: Histogram Clustering for Mixture of Markov Chains (No Actions)

This module implements the histogram clustering stage of the three-stage algorithm
for learning mixtures of Markov chains, adapted from the mdpmix approach but
simplified for the no-action scenario.

The key idea is to compute dissimilarity statistics between trajectories based on
their subspace projections, then use histogram-based thresholding and spectral
clustering to group trajectories into clusters.
"""

import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy import stats
from typing import List, Tuple, Dict, Optional
from utils import error_rate

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def compute_dissimilarity_statistic(trajectories, eigvecs, partitions, verbose=False):
    """
    Compute dissimilarity statistics between trajectories based on subspace projections.
    
    This is adapted from mdpmix/clustering.py but simplified for the no-action case.
    
    Args:
        trajectories: List of trajectories, each as list of states
        eigvecs: Eigenvectors from Stage 1 subspace estimation
        partitions: Trajectory partitions used in Stage 1
        verbose: Print progress information
        
    Returns:
        stat_matrix: Dissimilarity matrix of shape (n_trajectories, n_trajectories)
    """
    if verbose:
        print("Computing dissimilarity statistics...")
    
    n_trajectories = len(trajectories)
    n_states = eigvecs.shape[0]
    K = eigvecs.shape[2]
    
    omega_one, omega_two = partitions
    
    # Convert trajectories to one-hot encoding for both partitions
    onehot_states_1 = np.zeros((n_trajectories, len(omega_one), n_states))
    onehot_next_states_1 = np.zeros((n_trajectories, len(omega_one), n_states))
    
    onehot_states_2 = np.zeros((n_trajectories, len(omega_two), n_states))
    onehot_next_states_2 = np.zeros((n_trajectories, len(omega_two), n_states))
    
    for traj_idx, trajectory in enumerate(trajectories):
        for i, t in enumerate(omega_one):
            if t < len(trajectory) - 1:
                onehot_states_1[traj_idx, i, trajectory[t]] = 1
                onehot_next_states_1[traj_idx, i, trajectory[t+1]] = 1
        
        for i, t in enumerate(omega_two):
            if t < len(trajectory) - 1:
                onehot_states_2[traj_idx, i, trajectory[t]] = 1
                onehot_next_states_2[traj_idx, i, trajectory[t+1]] = 1
    
    # Compute empirical transition matrices for both partitions
    from mdpmix_stage1_subspace import get_empirical_transition_matrix
    
    h1, _ = get_empirical_transition_matrix(onehot_states_1, onehot_next_states_1)
    h2, _ = get_empirical_transition_matrix(onehot_states_2, onehot_next_states_2)
    
    # Project transition matrices to subspace
    # projs has shape (2, n_trajectories, n_states, K)
    projs = np.zeros((2, n_trajectories, n_states, K))
    
    for traj_idx in range(n_trajectories):
        for state_idx in range(n_states):
            # Project h1 and h2 to subspace
            projs[0, traj_idx, state_idx, :] = h1[traj_idx, state_idx, :] @ eigvecs[state_idx, :, :]
            projs[1, traj_idx, state_idx, :] = h2[traj_idx, state_idx, :] @ eigvecs[state_idx, :, :]
    
    if verbose:
        print(f"Computed projections of shape {projs.shape}")
    
    # Compute dissimilarity statistics
    # For each pair of trajectories, compute the maximum inner product difference
    stat_matrix = np.zeros((n_trajectories, n_trajectories))
    
    for i in tqdm(range(n_trajectories), desc="Computing dissimilarities", disable=not verbose):
        for j in range(n_trajectories):
            if i == j:
                continue
                
            # Compute difference in projections
            diff_proj1 = projs[0, i, :, :] - projs[0, j, :, :]  # Shape: (n_states, K)
            diff_proj2 = projs[1, i, :, :] - projs[1, j, :, :]  # Shape: (n_states, K)
            
            # Compute inner products and take maximum
            # Match mdpmix: use np.nansum to be robust to NaNs
            inner_products = np.nansum(diff_proj1 * diff_proj2, axis=1)  # Shape: (n_states,)
            stat_matrix[i, j] = np.nanmax(inner_products)
    
    if verbose:
        print(f"Dissimilarity matrix computed. Range: [{stat_matrix.min():.4f}, {stat_matrix.max():.4f}]")
    
    return stat_matrix

def compute_histogram_threshold(stat_matrix, K, percentile=50, verbose=False):
    """
    Compute threshold for clustering using histogram analysis.
    
    Args:
        stat_matrix: Dissimilarity matrix
        K: Number of clusters
        percentile: Percentile to use for threshold (default: median)
        verbose: Print progress information
        
    Returns:
        threshold: Clustering threshold
        histogram_data: Dictionary with histogram information
    """
    if verbose:
        print(f"Computing histogram threshold for K={K} clusters...")
    
    # Extract upper triangular values (excluding diagonal)
    n = stat_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    dissimilarities = stat_matrix[upper_tri_indices]
    
    # Compute threshold using percentile
    threshold = np.percentile(dissimilarities, percentile)
    
    # Create histogram data for analysis
    histogram_data = {
        'dissimilarities': dissimilarities,
        'threshold': threshold,
        'percentile': percentile,
        'mean': np.mean(dissimilarities),
        'std': np.std(dissimilarities),
        'min': np.min(dissimilarities),
        'max': np.max(dissimilarities)
    }
    
    if verbose:
        print(f"Threshold: {threshold:.4f} (percentile {percentile})")
        print(f"Dissimilarity range: [{histogram_data['min']:.4f}, {histogram_data['max']:.4f}]")
        print(f"Mean ± std: {histogram_data['mean']:.4f} ± {histogram_data['std']:.4f}")
    
    return threshold, histogram_data

def plot_dissimilarity_histogram(histogram_data, threshold, save_path=None, verbose=False):
    """
    Plot histogram of dissimilarity statistics with threshold line.
    
    Args:
        histogram_data: Dictionary with histogram information
        threshold: Clustering threshold
        save_path: Path to save the plot (optional)
        verbose: Print progress information
    """
    if verbose:
        print("Plotting dissimilarity histogram...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    dissimilarities = histogram_data['dissimilarities']
    plt.hist(dissimilarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.4f}')
    
    # Add statistics
    plt.axvline(x=histogram_data['mean'], color='green', linestyle='-', linewidth=1,
                label=f'Mean = {histogram_data["mean"]:.4f}')
    
    plt.xlabel('Dissimilarity Statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dissimilarity Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Histogram saved to {save_path}")
    
    plt.show()

def spectral_clustering_with_threshold(stat_matrix, threshold, K, method='kmeans', verbose=False):
    """
    Perform spectral clustering using the dissimilarity matrix and threshold.
    
    Args:
        stat_matrix: Dissimilarity matrix
        threshold: Clustering threshold
        K: Number of clusters
        method: Clustering method ('kmeans' or 'discretize')
        verbose: Print progress information
        
    Returns:
        cluster_labels: Array of cluster assignments
    """
    if verbose:
        print(f"Performing spectral clustering with threshold {threshold:.4f}...")
    
    # Create similarity matrix: trajectories are similar if dissimilarity < threshold
    similarity_matrix = (stat_matrix < threshold).astype(int)
    
    # Ensure diagonal is 1 (trajectory is similar to itself)
    np.fill_diagonal(similarity_matrix, 1)
    
    if verbose:
        n_connections = np.sum(similarity_matrix) - len(similarity_matrix)  # Exclude diagonal
        total_possible = len(similarity_matrix) * (len(similarity_matrix) - 1)
        connection_rate = n_connections / total_possible
        print(f"Connection rate: {connection_rate:.3f} ({n_connections}/{total_possible})")
    
    # Perform spectral clustering
    clustering = SpectralClustering(
        n_clusters=K,
        affinity='precomputed',
        assign_labels=method
    )
    
    cluster_labels = clustering.fit_predict(similarity_matrix)
    
    if verbose:
        print(f"Spectral clustering completed. Cluster sizes: {np.bincount(cluster_labels)}")
    
    return cluster_labels


def histogram_clustering_stage(trajectories, eigvecs, partitions, K, 
                             percentile=50, plot_histogram=True, verbose=False):
    """
    Main function for Stage 2: Histogram Clustering.
    
    This is the entry point for the second stage of the three-stage algorithm.
    
    Args:
        trajectories: List of trajectories, each as list of states
        eigvecs: Eigenvectors from Stage 1
        partitions: Trajectory partitions from Stage 1
        K: Number of mixture components
        percentile: Percentile for threshold computation
        plot_histogram: Whether to plot dissimilarity histogram
        verbose: Print progress information
        
    Returns:
        cluster_labels: Array of cluster assignments
        stat_matrix: Dissimilarity matrix
        threshold: Clustering threshold
        histogram_data: Dictionary with histogram information
    """
    if verbose:
        print(f"\nStage 2: Histogram Clustering")
        print(f"Number of trajectories: {len(trajectories)}")
        print(f"Number of components: {K}")
    
    # Step 1: Compute dissimilarity statistics
    stat_matrix = compute_dissimilarity_statistic(trajectories, eigvecs, partitions, verbose)
    
    # Step 2: Compute threshold using histogram analysis
    threshold, histogram_data = compute_histogram_threshold(
        stat_matrix, K, percentile=percentile, verbose=verbose
    )
    
    # Step 3: Plot histogram if requested
    if plot_histogram:
        plot_dissimilarity_histogram(
            histogram_data, threshold, 
            save_path='dissimilarity_histogram.png' if verbose else None,
            verbose=verbose
        )
    
    # Step 4: Perform spectral clustering
    cluster_labels = spectral_clustering_with_threshold(
        stat_matrix, threshold, K, verbose=verbose
    )
    
    if verbose:
        print(f"Stage 2 completed. Cluster assignments: {cluster_labels}")
    
    return cluster_labels, stat_matrix, threshold, histogram_data

def clustering_diagnostics(stat_matrix, K, true_labels=None, 
                          percentile_range=(10, 90), num_thresholds=20, verbose=False):
    """
    Perform clustering diagnostics by varying the threshold.
    
    Args:
        stat_matrix: Dissimilarity matrix
        K: Number of clusters
        true_labels: True cluster labels (optional)
        percentile_range: Range of percentiles to test
        num_thresholds: Number of threshold values to test
        verbose: Print progress information
        
    Returns:
        results: Dictionary with diagnostic results
    """
    if verbose:
        print("Running clustering diagnostics...")
    
    # Generate threshold range
    percentiles = np.linspace(percentile_range[0], percentile_range[1], num_thresholds)
    thresholds = [np.percentile(stat_matrix[np.triu_indices_from(stat_matrix, k=1)], p) 
                  for p in percentiles]
    
    accuracies = []
    cluster_counts = []
    
    for threshold in thresholds:
        # Perform clustering
        cluster_labels = spectral_clustering_with_threshold(stat_matrix, threshold, K)
        
        # Count unique clusters
        unique_clusters = len(np.unique(cluster_labels))
        cluster_counts.append(unique_clusters)
        
        # Compute accuracy if true labels available
        if true_labels is not None:
            accuracy = evaluate_clustering_accuracy(cluster_labels, true_labels)
            accuracies.append(accuracy)
        else:
            accuracies.append(np.nan)
    
    results = {
        'percentiles': percentiles,
        'thresholds': thresholds,
        'accuracies': np.array(accuracies),
        'cluster_counts': np.array(cluster_counts)
    }
    
    if verbose:
        if true_labels is not None:
            best_idx = np.nanargmax(accuracies)
            print(f"Best accuracy: {accuracies[best_idx]:.4f} at percentile {percentiles[best_idx]:.1f}")
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test with simple synthetic data
    np.random.seed(42)
    
    # Create simple synthetic trajectories (same as Stage 1 test)
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
    
    # Test Stage 1 first (needed for Stage 2)
    from mdpmix_stage1_subspace import subspace_estimation_stage
    
    print("Running Stage 1: Subspace Estimation...")
    eigvals, eigvecs, partitions = subspace_estimation_stage(
        trajectories, K=2, use_transitions=True, verbose=True
    )
    
    # Test Stage 2
    print("\nTesting Stage 2: Histogram Clustering...")
    cluster_labels, stat_matrix, threshold, histogram_data = histogram_clustering_stage(
        trajectories, eigvecs, partitions, K=2, verbose=True
    )
    
    # Evaluate accuracy
    accuracy = 1.0 - error_rate(cluster_labels, np.asarray(true_labels))
    
    print(f"\nResults:")
    print(f"Clustering accuracy: {accuracy:.4f}")
    print(f"Cluster assignments: {cluster_labels}")
    print(f"True labels: {true_labels}")
