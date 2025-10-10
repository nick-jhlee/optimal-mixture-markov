Codes for the following paper: Clustering in Mixtures of Ergodic Markov Chains: Fundamental Limit and Two-Stage Algorithmd.


# Install
Clone the repository and first run
```shell
$ conda env create -f environment.yml
```
to create a conda environment.

All figures and results are saved in the `results` folder.

# Reproducing Figure 1
Run
```shell
$ python main_synthetic.py
```

# Reproducing Figure 2
Run
```shell
$ python ablation1.py
```

# Reproducing Figure 3
Run
```shell
$ python ablation2.py
```

# Miscellaneous Implementation Details

## Environment Configuration
see `Synthetic.py` for the environment configuration.

## Our Two-Stage Algorithm
see `Clustering.py` for the implementation of our two-stage algorithm.


## Algorithm of Kausik et al. (2023)
see `mdpmix_three_stage.py` for the implementation of the algorithm of Kausik et al. (2023).

This is divided into three stages:
- Stage 1: Subspace estimation via trajectory partitioning (`mdpmix_stage1_subspace.py`)
- Stage 2: Histogram-based thresholding and spectral clustering (`mdpmix_stage2_clustering.py`)
- Stage 3: EM refinement of transition matrices and priors (`mdpmix_stage3_em.py`)

Credits:
- Based on the original mdpmix implementation: [github.com/chinmayakausik/mdpmix](https://github.com/chinmayakausik/mdpmix)
- Paper: C. Kausik, K. Tan, A. Tewari, "Learning Mixtures of Markov Chains and MDPs," ICML 2023. [PMLR link](https://proceedings.mlr.press/v202/kausik23a.html)
