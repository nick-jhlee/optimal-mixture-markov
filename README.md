## Optimal Mixture of Markov Chains (Three-Stage)

Minimal implementation of a three-stage algorithm for learning mixtures of Markov chains (no actions):
- Stage 1: Subspace estimation via trajectory partitioning
- Stage 2: Histogram-based thresholding and spectral clustering
- Stage 3: EM refinement of transition matrices and priors

Quick start: see `main_synthetic.py` or `main_lastfm.py`.

Credits:
- Based on the original mdpmix implementation: [github.com/chinmayakausik/mdpmix](https://github.com/chinmayakausik/mdpmix)
- Paper: C. Kausik, K. Tan, A. Tewari, "Learning Mixtures of Markov Chains and MDPs," ICML 2023. [PMLR link](https://proceedings.mlr.press/v202/kausik23a.html)

License: See `LICENSE`.
