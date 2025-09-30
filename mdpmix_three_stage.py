"""
Three-Stage Algorithm for Learning Mixtures of Markov Chains (No Actions)

This module provides a unified interface for the complete three-stage algorithm
for learning mixtures of Markov chains, adapted from the mdpmix approach.

The three stages are:
1. Subspace Estimation: Estimate the K-dimensional subspace using trajectory partitioning
2. Histogram Clustering: Use dissimilarity statistics and spectral clustering
3. EM Algorithm: Refine estimates using expectation-maximization

This implementation is specifically designed for the no-action scenario (Markov chains
instead of MDPs) and provides a clean, easy-to-use interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import warnings
from tqdm import tqdm

# Import the three stages
from mdpmix_stage1_subspace import subspace_estimation_stage
from mdpmix_stage2_clustering import histogram_clustering_stage
from utils import error_rate
from mdpmix_stage3_em import em_stage

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MixtureMarkovChainLearner:
    """
    Main class for learning mixtures of Markov chains using the three-stage algorithm.
    
    This class provides a unified interface to the complete algorithm, handling
    all three stages internally and providing convenient methods for training,
    evaluation, and prediction.
    """
    
    def __init__(self, K: int, n_states: Optional[int] = None, 
                 use_transitions: bool = True, hard_em: bool = True,
                 verbose: bool = True):
        """
        Initialize the mixture Markov chain learner.
        
        Args:
            K: Number of mixture components
            n_states: Number of states (inferred if None)
            use_transitions: Whether to use transition-based or occupancy-based estimation
            hard_em: Whether to use hard assignments in EM algorithm
            verbose: Whether to print progress information
        """
        self.K = K
        self.n_states = n_states
        self.use_transitions = use_transitions
        self.hard_em = hard_em
        self.verbose = verbose
        
        # Results from each stage
        self.stage1_results = None
        self.stage2_results = None
        self.stage3_results = None
        
        # Final models
        self.transition_matrices = None
        self.cluster_priors = None
        self.start_probs = None
        self.cluster_labels = None
        
        # Training metadata
        self.trained = False
        self.training_history = {}
    
    def fit(self, trajectories: List[List[int]], 
            percentile: int = 50, max_em_iterations: int = 100,
            plot_histogram: bool = False) -> Dict:
        """
        Train the mixture Markov chain learner on the given trajectories.
        
        Args:
            trajectories: List of trajectories, each as list of states
            percentile: Percentile for threshold computation in Stage 2
            max_em_iterations: Maximum number of EM iterations
            plot_histogram: Whether to plot dissimilarity histogram
            
        Returns:
            Dictionary containing training results and metadata
        """
        if self.verbose:
            print("=" * 60)
            print("THREE-STAGE MIXTURE MARKOV CHAIN LEARNING")
            print("=" * 60)
        
        # Infer number of states if not provided
        if self.n_states is None:
            self.n_states = max(max(traj) for traj in trajectories) + 1
        
        if self.verbose:
            print(f"Training on {len(trajectories)} trajectories")
            print(f"Number of states: {self.n_states}")
            print(f"Number of components: {self.K}")
        
        # Stage 1: Subspace Estimation
        if self.verbose:
            print("\n" + "-" * 40)
            print("STAGE 1: SUBSPACE ESTIMATION")
            print("-" * 40)
        
        try:
            eigvals, eigvecs, partitions = subspace_estimation_stage(
                trajectories, self.K, use_transitions=self.use_transitions, 
                verbose=self.verbose
            )
            
            self.stage1_results = {
                'eigvals': eigvals,
                'eigvecs': eigvecs,
                'partitions': partitions
            }
            
            if self.verbose:
                print("✓ Stage 1 completed successfully")
                
        except Exception as e:
            if self.verbose:
                print(f"✗ Stage 1 failed: {e}")
            raise
        
        # Stage 2: Histogram Clustering
        if self.verbose:
            print("\n" + "-" * 40)
            print("STAGE 2: HISTOGRAM CLUSTERING")
            print("-" * 40)
        
        try:
            cluster_labels, stat_matrix, threshold, histogram_data = histogram_clustering_stage(
                trajectories, eigvecs, partitions, self.K,
                percentile=percentile, plot_histogram=plot_histogram,
                verbose=self.verbose
            )
            
            self.stage2_results = {
                'cluster_labels': cluster_labels,
                'stat_matrix': stat_matrix,
                'threshold': threshold,
                'histogram_data': histogram_data
            }
            
            if self.verbose:
                print("✓ Stage 2 completed successfully")
                
        except Exception as e:
            if self.verbose:
                print(f"✗ Stage 2 failed: {e}")
            raise
        
        # Stage 3: EM Algorithm
        if self.verbose:
            print("\n" + "-" * 40)
            print("STAGE 3: EM ALGORITHM")
            print("-" * 40)
        
        try:
            em_results = em_stage(
                trajectories, cluster_labels, self.K, self.n_states,
                max_iterations=max_em_iterations, hard_assignments=self.hard_em,
                verbose=self.verbose
            )
            
            self.stage3_results = em_results
            
            # Extract final models
            self.transition_matrices = em_results['transition_matrices']
            self.cluster_priors = em_results['cluster_priors']
            self.start_probs = em_results['start_probs']
            self.cluster_labels = em_results['cluster_labels']
            
            if self.verbose:
                print("✓ Stage 3 completed successfully")
                
        except Exception as e:
            if self.verbose:
                print(f"✗ Stage 3 failed: {e}")
            raise
        
        # Training completed
        self.trained = True
        
        # Store training history
        self.training_history = {
            'stage1': self.stage1_results,
            'stage2': self.stage2_results,
            'stage3': self.stage3_results,
            'parameters': {
                'K': self.K,
                'n_states': self.n_states,
                'use_transitions': self.use_transitions,
                'hard_em': self.hard_em,
                'percentile': percentile,
                'max_em_iterations': max_em_iterations
            }
        }
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Final cluster sizes: {np.bincount(self.cluster_labels)}")
            print(f"Cluster priors: {self.cluster_priors}")
            print(f"EM converged: {em_results['converged']}")
            print(f"EM iterations: {em_results['iterations']}")
        
        return self.training_history
    
    def predict(self, trajectories: List[List[int]]) -> np.ndarray:
        """
        Predict cluster assignments for new trajectories.
        
        Args:
            trajectories: List of trajectories to classify
            
        Returns:
            Array of cluster assignments
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        cluster_assignments = np.zeros(len(trajectories), dtype=int)
        
        for i, trajectory in enumerate(trajectories):
            best_cluster = 0
            best_likelihood = float('-inf')
            
            for k in range(self.K):
                # Compute log-likelihood under cluster k
                log_likelihood = 0.0
                
                # Starting state probability
                if self.start_probs is not None:
                    log_likelihood += np.log(self.start_probs[k, trajectory[0]] + 1e-10)
                
                # Transition probabilities
                for t in range(len(trajectory) - 1):
                    current_state = trajectory[t]
                    next_state = trajectory[t + 1]
                    prob = self.transition_matrices[k, current_state, next_state]
                    log_likelihood += np.log(prob + 1e-10)
                
                # Add log prior
                log_likelihood += np.log(self.cluster_priors[k] + 1e-10)
                
                if log_likelihood > best_likelihood:
                    best_likelihood = log_likelihood
                    best_cluster = k
            
            cluster_assignments[i] = best_cluster
        
        return cluster_assignments
    
    def predict_proba(self, trajectories: List[List[int]]) -> np.ndarray:
        """
        Predict cluster assignment probabilities for new trajectories.
        
        Args:
            trajectories: List of trajectories to classify
            
        Returns:
            Array of shape (n_trajectories, K) with assignment probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = np.zeros((len(trajectories), self.K))
        
        for i, trajectory in enumerate(trajectories):
            log_likelihoods = np.zeros(self.K)
            
            for k in range(self.K):
                # Compute log-likelihood under cluster k
                log_likelihood = 0.0
                
                # Starting state probability
                if self.start_probs is not None:
                    log_likelihood += np.log(self.start_probs[k, trajectory[0]] + 1e-10)
                
                # Transition probabilities
                for t in range(len(trajectory) - 1):
                    current_state = trajectory[t]
                    next_state = trajectory[t + 1]
                    prob = self.transition_matrices[k, current_state, next_state]
                    log_likelihood += np.log(prob + 1e-10)
                
                # Add log prior
                log_likelihood += np.log(self.cluster_priors[k] + 1e-10)
                log_likelihoods[k] = log_likelihood
            
            # Convert to probabilities using softmax
            max_ll = np.max(log_likelihoods)
            exp_ll = np.exp(log_likelihoods - max_ll)
            probabilities[i, :] = exp_ll / np.sum(exp_ll)
        
        return probabilities
    
    def evaluate(self, trajectories: List[List[int]], true_labels: List[int]) -> Dict:
        """
        Evaluate the model on test trajectories.
        
        Args:
            trajectories: List of test trajectories
            true_labels: True cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        predicted_labels = self.predict(trajectories)
        accuracy = 1.0 - error_rate(predicted_labels, true_labels)
        
        return {
            'accuracy': accuracy,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        }
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot EM convergence if available.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.trained or self.stage3_results is None:
            raise ValueError("Model must be trained to plot convergence")
        
        log_likelihoods = self.stage3_results['log_likelihoods']
        
        plt.figure(figsize=(10, 6))
        plt.plot(log_likelihoods, 'b-', linewidth=2)
        plt.xlabel('EM Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('EM Algorithm Convergence')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def get_models(self) -> Dict:
        """
        Get the learned models.
        
        Returns:
            Dictionary containing transition matrices, priors, and other models
        """
        if not self.trained:
            raise ValueError("Model must be trained to get models")
        
        return {
            'transition_matrices': self.transition_matrices,
            'cluster_priors': self.cluster_priors,
            'start_probs': self.start_probs,
            'cluster_labels': self.cluster_labels,
            'K': self.K,
            'n_states': self.n_states
        }
    
    def save_results(self, filepath: str):
        """
        Save training results to file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.trained:
            raise ValueError("Model must be trained to save results")
        
        np.savez(filepath, **{
            'transition_matrices': self.transition_matrices,
            'cluster_priors': self.cluster_priors,
            'start_probs': self.start_probs,
            'cluster_labels': self.cluster_labels,
            'training_history': self.training_history,
            'K': self.K,
            'n_states': self.n_states
        })
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load training results from file.
        
        Args:
            filepath: Path to load the results from
        """
        data = np.load(filepath, allow_pickle=True)
        
        self.transition_matrices = data['transition_matrices']
        self.cluster_priors = data['cluster_priors']
        self.start_probs = data['start_probs']
        self.cluster_labels = data['cluster_labels']
        self.training_history = data['training_history'].item()
        self.K = int(data['K'])
        self.n_states = int(data['n_states'])
        self.trained = True
        
        print(f"Results loaded from {filepath}")

def run_complete_algorithm(trajectories: List[List[int]], K: int, 
                          true_labels: Optional[List[int]] = None,
                          **kwargs) -> Tuple[MixtureMarkovChainLearner, Dict]:
    """
    Convenience function to run the complete three-stage algorithm.
    
    Args:
        trajectories: List of trajectories
        K: Number of mixture components
        true_labels: True cluster labels (optional, for evaluation)
        **kwargs: Additional arguments passed to fit()
        
    Returns:
        Tuple of (trained_learner, evaluation_results)
    """
    # Create and train learner
    learner = MixtureMarkovChainLearner(K=K, verbose=True)
    training_results = learner.fit(trajectories, **kwargs)
    
    # Evaluate if true labels provided
    evaluation_results = None
    if true_labels is not None:
        evaluation_results = learner.evaluate(trajectories, true_labels)
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    
    return learner, evaluation_results

# Example usage and testing
if __name__ == "__main__":
    # Test with simple synthetic data
    np.random.seed(42)
    
    # Create simple synthetic trajectories
    n_trajectories = 100
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
    
    # Test complete algorithm
    print("Testing Complete Three-Stage Algorithm...")
    learner, eval_results = run_complete_algorithm(
        trajectories, K=2, true_labels=true_labels,
        max_em_iterations=50, plot_histogram=True
    )
    
    # Test prediction on new trajectories
    print("\nTesting prediction on new trajectories...")
    new_trajectories = trajectories[:10]  # Use first 10 as "new" data
    predictions = learner.predict(new_trajectories)
    probabilities = learner.predict_proba(new_trajectories)
    
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Plot convergence
    learner.plot_convergence(save_path='em_convergence.png')
    
    print("\nTest completed successfully!")
