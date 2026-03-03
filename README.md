Codes for [_Near-Optimal Clustering in Mixture of Markov Chains_](https://arxiv.org/abs/2506.01324) (AISTATS 2026) by [Junghyun Lee](https://nick-jhlee.github.io/), [Yassir Jedra](https://sites.google.com/view/yassir-jedra/home), [Alexandre Proutière](https://people.kth.se/~alepro/), and [Se-Young Yun](https://fbsqkd.github.io/).

If you plan to use this repository or cite our paper, please use the following bibtex format:

```latex
@InProceedings{lee2026markov,
  title = 	 {{Near-Optimal Clustering in Mixture of Markov Chains}},
  author =       {Lee, Junghyun and Jedra, Yassir and Proutière, Alexandre and Yun, Se-Young},
  booktitle = 	 {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  year = 	 {2026},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--05 May},
  publisher =    {PMLR},
  url = 	 {https://arxiv.org/abs/2506.01324},
}
```



# Install

## Option 1: Automated Setup (Recommended)
Clone the repository and run the setup script:
```shell
$ ./setup_env.sh
```
This script will:
- Install `uv` if not already installed
- Create a virtual environment named `markov`
- Install all dependencies to their latest versions

After setup, activate the environment:
```shell
$ source markov/bin/activate
```

## Option 2: Using pip
If you prefer using traditional pip:
```shell
$ python -m venv markov
$ source markov/bin/activate
$ pip install -r requirements.txt
```

## Option 3: Manual Setup with uv
If you already have `uv` installed:
```shell
$ uv venv markov --python 3.11
$ source markov/bin/activate
$ uv pip install numpy scipy pandas matplotlib scikit-learn numba joblib tqdm cloudpickle gymnasium farama-notifications
```

All figures and results are saved in the `results` folder.

# Reproducing Figure 1 (Main Figure in Section 6)
Run
```shell
$ python main_synthetic.py
```

# Reproducing Figure 2 (Ablation #1 in Appendix F.2)
Run
```shell
$ python ablation1.py
```

# Reproducing Figure 3 (Ablation #2 in Appendix F.3)
Run
```shell
$ python ablation2.py
```

# Reproducing Figure 4 (Ablation #3 in Appendix F.4)
Run
```shell
$ python ablation4.py
```

# Reproducing Figure 5 (Ablation #4 in Appendix F.5)
Run
```shell
$ python ablation5.py
```

# Reproducing Figure 6 (Last.fm 1K Experiment in Appendix F.6)
Run
```shell
$ python main_lastfm1k.py
```
The raw datasets are downloaded from:
- Lastfm-dataset-1K: http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
- Lastfm-ArtistTags2007: https://web.archive.org/web/20110827230919/http://static.echonest.com/Lastfm-ArtistTags2007.tar.gz


# Miscellaneous Implementation Details

## Environment Configuration
see `Synthetic.py` for the environment configuration.

## Our Two-Stage Algorithm
see `Clustering.py` for the implementation of our two-stage algorithm.


## Algorithm of Kausik et al. (2023)
see the folder `mcmix` for the implementation of the algorithm of Kausik et al. (2023).



Credits:
- Based on the original mdpmix implementation: [github.com/chinmayakausik/mdpmix](https://github.com/chinmayakausik/mdpmix)
- Paper: C. Kausik, K. Tan, A. Tewari, "Learning Mixtures of Markov Chains and MDPs," ICML 2023. [PMLR link](https://proceedings.mlr.press/v202/kausik23a.html)



