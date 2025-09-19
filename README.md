# Clustering in Mixture of Ergodic Markov Chains: Fundamental Limits and a Two-Stage Algorithm

This repository implements a two-stage clustering algorithm for mixture Markov chains with parallelized evaluation and statistical analysis.

## Overview

The algorithm performs clustering on mixture Markov chains using:
1. **Stage I**: Initial spectral clustering
2. **Stage II**: Likelihood-based refinement (1 iteration)
3. **Stage II**: Likelihood-based refinement (10 iterations)

## Features

- **Parallelized execution** using joblib for efficient computation
- **Statistical robustness** with 30 repeats per configuration and bootstrap confidence intervals
- **Comprehensive evaluation** across multiple T (trajectories) and H (trajectory length) parameters
- **Visualization** with 2D error rate plots and 3D surface plots
- **Error bars** showing 95% confidence intervals

## Files

- `main.py` - Main execution script with parallelized evaluation
- `Synthetic.py` - Synthetic data generation and Markov chain utilities
- `Clustering.py` - Clustering algorithms (spectral and likelihood-based)
- `utils.py` - Utility functions (error rate calculation, KL divergence)

## Usage

```python
# Run the experiment
python main.py
```

The script will:
1. Generate synthetic mixture Markov chain data
2. Run clustering algorithms across T and H parameter ranges
3. Compute bootstrap confidence intervals
4. Generate plots and save results to `results.csv`

## Parameters

- **T_list**: Number of trajectories [100, 200, 300, 400, 500, 600]
- **H_list**: Trajectory length [100, 200, ..., 2000]
- **n_repeat**: Number of repeats per configuration (default: 30)
- **alpha**: Confidence level for bootstrap CIs (default: 0.05)
- **n_jobs**: Number of parallel jobs (default: -1, uses all cores)

## Output

- **results.csv**: Detailed results with means and confidence intervals
- **2D plots**: Error rate vs H for each T value with error bars
- **3D surface plots**: Interpolated error surfaces for all stages

## Algorithm Details

The clustering process involves:
1. Generating synthetic trajectories from mixture Markov chains
2. Applying spectral clustering for initial assignment
3. Refining assignments using likelihood-based optimization
4. Computing error rates and statistical confidence intervals

## Dependencies

- numpy
- matplotlib
- scipy
- joblib
- Custom modules: Synthetic, Clustering, utils
