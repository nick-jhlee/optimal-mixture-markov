from utils import *
from Synthetic import *
from Clustering import *

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def process_single_config(T, H, K, S, delta, n_repeat=30):
    """Process a single (T, H) configuration with multiple repeats - designed for parallel execution"""
    try:
        stage_i_errors = []
        stage_ii_errors = []
        stage_ii_errors_10 = []
        
        for _ in range(n_repeat):
            env_config = {
                'H': H,
                'K': K,
                'S': S
            }
            env = MixtureMarkovChains(env_config)
            gamma_ps = env.gamma_ps
            f, trajectories = env.generate_trajectories(T)

            # Stage I: Initial Spectral Clustering
            f_hat_1 = InitialSpectral(trajectories, T, H, S, gamma_ps, delta)
            stage_i_error = error_rate(f, f_hat_1)
            stage_i_errors.append(stage_i_error)
            
            # Stage II: Likelihood-based Refinement (1 iteration)
            f_hat_2 = LikelihoodRefinement(trajectories, f_hat_1, T, S)
            stage_ii_error = error_rate(f, f_hat_2)
            stage_ii_errors.append(stage_ii_error)
            
            # Stage II: Likelihood-based Refinement (10 iterations)
            f_hat_2_10 = LikelihoodRefinement(trajectories, f_hat_1, T, S, 10)
            stage_ii_error_10 = error_rate(f, f_hat_2_10)
            stage_ii_errors_10.append(stage_ii_error_10)
        
        return T, H, stage_i_errors, stage_ii_errors, stage_ii_errors_10
        
    except Exception as e:
        print(f"Error processing T={T}, H={H}: {e}")
        return T, H, [float('inf')] * n_repeat, [float('inf')] * n_repeat, [float('inf')] * n_repeat

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

def vary_T_H(T_list, H_list, K, S, delta, n_jobs=-1, n_repeat=30, alpha=0.05):
    """Parallelized version of vary_T_H using joblib with repeats and error bars"""
    print(f"Processing {len(T_list)} × {len(H_list)} = {len(T_list) * len(H_list)} configurations...")
    print(f"Using {n_jobs} parallel jobs, {n_repeat} repeats per configuration")
    
    # Create all (T, H) combinations
    combinations = [(T, H) for T in T_list for H in H_list]
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_config)(T, H, K, S, delta, n_repeat) 
        for T, H in combinations
    )
    
    # Unpack results
    T_results, H_results, stage_i_errors_list, stage_ii_errors_list, stage_ii_errors_10_list = zip(*results)
    
    # Convert to numpy arrays for easier handling
    T_results = np.array(T_results)
    H_results = np.array(H_results)
    
    # Convert lists to numpy arrays for easier handling
    stage_i_errors_array = np.array(stage_i_errors_list)  # shape: (n_configs, n_repeat)
    stage_ii_errors_array = np.array(stage_ii_errors_list)  # shape: (n_configs, n_repeat)
    stage_ii_10_errors_array = np.array(stage_ii_errors_10_list)  # shape: (n_configs, n_repeat)
    
    # Compute means and confidence intervals for each (T, H) pair
    stage_i_means = []
    stage_i_errors_lower = []
    stage_i_errors_upper = []
    
    stage_ii_means = []
    stage_ii_errors_lower = []
    stage_ii_errors_upper = []
    
    stage_ii_10_means = []
    stage_ii_10_errors_lower = []
    stage_ii_10_errors_upper = []
    
    for i in range(len(T_results)):
        # Stack the three stages for this (T, H) pair: shape (n_repeat, 3)
        three_stages = np.column_stack((
            stage_i_errors_array[i],      # Stage I
            stage_ii_errors_array[i],     # Stage II (1 iter)
            stage_ii_10_errors_array[i]   # Stage II (10 iter)
        ))
        
        # Compute CI for all 3 stages together for this (T, H) pair
        means, ci_lower, ci_upper = bootstrap_mean_ci_multi(three_stages, alpha=alpha)
        
        # Extract results for each stage
        stage_i_means.append(means[0])
        stage_i_errors_lower.append(means[0] - ci_lower[0])
        stage_i_errors_upper.append(ci_upper[0] - means[0])
        
        stage_ii_means.append(means[1])
        stage_ii_errors_lower.append(means[1] - ci_lower[1])
        stage_ii_errors_upper.append(ci_upper[1] - means[1])
        
        stage_ii_10_means.append(means[2])
        stage_ii_10_errors_lower.append(means[2] - ci_lower[2])
        stage_ii_10_errors_upper.append(ci_upper[2] - means[2])
    
    # Convert to numpy arrays
    stage_i_means = np.array(stage_i_means)
    stage_i_errors_lower = np.array(stage_i_errors_lower)
    stage_i_errors_upper = np.array(stage_i_errors_upper)
    
    stage_ii_means = np.array(stage_ii_means)
    stage_ii_errors_lower = np.array(stage_ii_errors_lower)
    stage_ii_errors_upper = np.array(stage_ii_errors_upper)
    
    stage_ii_10_means = np.array(stage_ii_10_means)
    stage_ii_10_errors_lower = np.array(stage_ii_10_errors_lower)
    stage_ii_10_errors_upper = np.array(stage_ii_10_errors_upper)

    # Save the results to a csv file
    results_data = np.column_stack((
        T_results, H_results, 
        stage_i_means, stage_i_errors_lower, stage_i_errors_upper,
        stage_ii_means, stage_ii_errors_lower, stage_ii_errors_upper,
        stage_ii_10_means, stage_ii_10_errors_lower, stage_ii_10_errors_upper
    ))
    np.savetxt('results.csv', results_data, delimiter=',', 
               header='T,H,StageI_mean,StageI_lower,StageI_upper,StageII_mean,StageII_lower,StageII_upper,StageII10_mean,StageII10_lower,StageII10_upper')
    
    # Create 2D plots of error rate vs H for each T value with error bars
    plot_2d_error_vs_H(T_results, H_results, stage_i_means, stage_ii_means, stage_ii_10_means,
                       stage_i_errors_lower, stage_i_errors_upper,
                       stage_ii_errors_lower, stage_ii_errors_upper,
                       stage_ii_10_errors_lower, stage_ii_10_errors_upper)
    
    return stage_i_means, stage_ii_means, stage_ii_10_means


def plot_3d_surfaces(T_results, H_results, stage_i_errors, stage_ii_errors, stage_ii_errors_10):
    """Create 3D interpolated surface plot with all stages combined"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define stages in reverse order for proper layering (back to front)
    stages = [
        (stage_ii_errors_10, 'green', 'Stage II (10 iter)'),  # Furthest back
        (stage_ii_errors, 'blue', 'Stage II (1 iter)'),     # Middle
        (stage_i_errors, 'red', 'Stage I')                  # Front
    ]
    
    for i, (errors, color, label) in enumerate(stages):
        # Create regular grid for interpolation
        T_grid = np.linspace(T_results.min(), T_results.max(), 30)
        H_grid = np.linspace(H_results.min(), H_results.max(), 30)
        T_mesh, H_mesh = np.meshgrid(T_grid, H_grid)
        
        # Interpolate surface
        points = np.column_stack((T_results, H_results))
        surface = griddata(points, errors, (T_mesh, H_mesh), method='cubic', fill_value=np.nan)
        
        # Plot surface with decreasing alpha for layering effect
        alpha_values = [0.3, 0.5, 0.7]  # Back to front: more transparent to less transparent
        ax.plot_surface(T_mesh, H_mesh, surface, alpha=alpha_values[i], color=color, label=label)
        
        # Plot original data points with decreasing alpha
        point_alpha = [0.6, 0.8, 1.0]  # Back to front: more transparent to opaque
        ax.scatter(T_results, H_results, errors, c=color, s=30, alpha=point_alpha[i])
    
    ax.set_xlabel('T (Number of Trajectories)')
    ax.set_ylabel('H (Trajectory Length)')
    ax.set_zlabel('Error Rate')
    ax.set_title('Error Rates vs T and H - All Stages Combined')
    ax.legend()
    
    # Set viewing angle so both T and H increase towards perspective
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()


def plot_2d_error_vs_H(T_results, H_results, stage_i_means, stage_ii_means, stage_ii_10_means,
                       stage_i_errors_lower, stage_i_errors_upper,
                       stage_ii_errors_lower, stage_ii_errors_upper,
                       stage_ii_10_errors_lower, stage_ii_10_errors_upper):
    """Create 2D plots of error rate vs H for each T value with error bars"""
    
    # Get unique T values and sort them
    unique_T = sorted(set(T_results))
    
    # Create subplots - one for each T value
    n_plots = len(unique_T)
    cols = min(3, n_plots)  # Max 3 columns
    rows = (n_plots + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Ensure axes is always a list of Axes objects
    axes = list(axes)
    
    # Color coding for stages
    stage_colors = ['red', 'blue', 'green']
    stage_labels = ['Stage I', 'Stage II (1 iter)', 'Stage II (10 iter)']
    
    for i, T_val in enumerate(unique_T):
        ax = axes[i]
        
        # Filter data for this T value
        mask = T_results == T_val
        H_vals = H_results[mask]
        stage_i_vals = stage_i_means[mask]
        stage_ii_vals = stage_ii_means[mask]
        stage_ii_10_vals = stage_ii_10_means[mask]
        
        # Error bars
        stage_i_lower = stage_i_errors_lower[mask]
        stage_i_upper = stage_i_errors_upper[mask]
        stage_ii_lower = stage_ii_errors_lower[mask]
        stage_ii_upper = stage_ii_errors_upper[mask]
        stage_ii_10_lower = stage_ii_10_errors_lower[mask]
        stage_ii_10_upper = stage_ii_10_errors_upper[mask]
        
        # Sort by H for proper line plotting
        sort_idx = np.argsort(H_vals)
        H_vals = H_vals[sort_idx]
        stage_i_vals = stage_i_vals[sort_idx]
        stage_ii_vals = stage_ii_vals[sort_idx]
        stage_ii_10_vals = stage_ii_10_vals[sort_idx]
        stage_i_lower = stage_i_lower[sort_idx]
        stage_i_upper = stage_i_upper[sort_idx]
        stage_ii_lower = stage_ii_lower[sort_idx]
        stage_ii_upper = stage_ii_upper[sort_idx]
        stage_ii_10_lower = stage_ii_10_lower[sort_idx]
        stage_ii_10_upper = stage_ii_10_upper[sort_idx]
        
        # Plot each stage with error bars
        ax.errorbar(H_vals, stage_i_vals, yerr=[stage_i_lower, stage_i_upper], 
                   fmt='o-', color=stage_colors[0], label=stage_labels[0], 
                   linewidth=2, markersize=6, capsize=5, capthick=2)
        ax.errorbar(H_vals, stage_ii_vals, yerr=[stage_ii_lower, stage_ii_upper], 
                   fmt='s-', color=stage_colors[1], label=stage_labels[1], 
                   linewidth=2, markersize=6, capsize=5, capthick=2)
        ax.errorbar(H_vals, stage_ii_10_vals, yerr=[stage_ii_10_lower, stage_ii_10_upper], 
                   fmt='^-', color=stage_colors[2], label=stage_labels[2], 
                   linewidth=2, markersize=6, capsize=5, capthick=2)
        
        ax.set_xlabel('H (Trajectory Length)')
        ax.set_ylabel('Error Rate')
        ax.set_title(f'T = {T_val} (Number of Trajectories)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def test(env_config, T, H, S, delta):
    """Test the clustering algorithm on a single configuration"""
    env = MixtureMarkovChains(env_config)
    print(f"Divergence: {env.D}")
    gamma_ps = env.gamma_ps
    f, trajectories = env.generate_trajectories(T)

    # Stage I: Initial Spectral Clustering
    f_hat_1 = InitialSpectral(trajectories, T, H, S, gamma_ps, delta)
    print(f"Stage I Error Rate: {error_rate(f, f_hat_1)}")

    # Stage II: Likelihood-based Refinement (1 iteration)
    f_hat_2 = LikelihoodRefinement(trajectories, f_hat_1, T, S)
    print(f"Stage II (1 iter) Error Rate: {error_rate(f, f_hat_2)}")

    # Stage II: Likelihood-based Refinement (10 iterations)
    f_hat_2_10 = LikelihoodRefinement(trajectories, f_hat_1, T, S, 10)
    print(f"Stage II (10 iter) Error Rate: {error_rate(f, f_hat_2_10)}")

if __name__ == '__main__':
    # Ground-truth, Synthetic MCC
    T, H, K, delta = 100, int(150), 2, 0.05
    S = 10
    
    # Define state partitions
    S_p, S_n = range(5), range(5, 10)
    
    # Create transition matrices
    P1 = np.zeros((S, S))
    P2 = np.zeros((S, S))
    
    for s in range(S):
        for s_ in range(S):
            if s_ in S_p:
                P1[s, s_] = 3 / S
                P2[s, s_] = 1 / S
            else:
                P1[s, s_] = 1 / S
                P2[s, s_] = 3 / S
    
    # Environment configuration
    env_config = {
        'H': H,
        'K': K,
        'S': S,
        'Ps': [P1, P2],
        'mus': [np.ones(S) / S, np.ones(S) / S]
    }

    # # Test single configuration
    # test(env_config, T, H, S, delta)

    # Full experiment
    print("\nRunning full experiment...")
    T_list = [100, 200, 300, 400, 500, 600]
    H_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
              1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    vary_T_H(T_list, H_list, K, S, delta, n_jobs=-1, n_repeat=30, alpha=0.05)