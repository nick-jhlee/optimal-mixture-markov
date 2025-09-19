# from https://github.com/nick-jhlee/optimal-block-mdp/blob/main/Synthetic.py
# modified by Junghyun Lee

import math
import numpy as np
import numpy.linalg as LA
from itertools import product
from utils import *

import gymnasium as gym


def spectral_gap(P, i):
    eigs, _ = LA.eigh(LA.matrix_power(P.T, i) @ LA.matrix_power(P, i))
    eigs = np.sort(eigs)
    # assert eigs[-1] == 1
    return eigs[-1] - eigs[-2]

def random_transition_matrices(K, S):
    Ps = []
    seeding_p = np.random.RandomState(seed=10)  # "local" seeding to fix the transition probability matrices
    for _ in range(K):
        p = seeding_p.rand(S, S)
        p /= p.sum(axis=1)[:, None]
        Ps.append(p)
    return Ps

def random_initial_distributions(K, S):
    mus = []
    seeding_m = np.random.RandomState(seed=20)
    for _ in range(K):
        m = seeding_m.rand(S)
        m /= m.sum()
        mus.append(m)
    return mus


class MixtureMarkovChains():
    f"""
    Synthetic Mixture of Markov Chains
    H, K, S, Ps, mus : change via env_config
    """

    def __init__(self, env_config={}):
        self.initialized = True
        params = env_config.keys()

        # Horizon length
        if 'H' in params:
            self.H = int(env_config['H'])
        else:
            self.H = 100

        # State space
        if 'S' in params:
            self.S = int(env_config['S'])
        else:
            self.S = 10
        
        # Number of Markov Chain Models
        if 'K' in params:
            self.K = int(env_config['K'])
        else:
            self.K = 10
        
        # Transition probability matrices
        if 'Ps' in params:
            self.Ps = env_config['Ps']
            if len(self.Ps) != self.K:
                raise Warning("Number of transition probability matrices must be equal to the number of Markov Chain Models... Regenerating the transition probability matrices")
                self.Ps = random_transition_matrices(self.K, self.S)
        else:
            self.Ps = random_transition_matrices(self.K, self.S)
        
        # Initial state distributions
        if 'mus' in params:
            self.mus = env_config['mus']
            if len(self.mus) != self.K:
                raise Warning("Number of initial state distributions must be equal to the number of Markov Chain Models... Regenerating the initial state distributions")
                self.mus = random_initial_distributions(self.K, self.S)
        else:
            self.mus = random_initial_distributions(self.K, self.S)

        # Initialize K environments, compute pseudo-spectral gaps
        self.envs = [MarkovChain(self.H, self.S, P, mu) for P, mu in zip(self.Ps, self.mus)]
        self.gamma_ps = min([env.gamma_ps() for env in self.envs])
        self.D = self.compute_divergence()
        
    def generate_trajectories(self, T, alphas=None):
        """
        Generate T trajectories of length H with prescribed cluster sizes.
        Default cluster sizes are all the same.
        """
        if alphas is None:
            alphas = [1/self.K for _ in range(self.K)]
        if len(alphas) != self.K:
            raise Warning("Number of alphas must be equal to the number of Markov Chain Models... Regenerating the alphas with uniform distribution")
            alphas = [1/self.K for _ in range(self.K)]

        trajectories = []
        f = {}
        cnt = 0
        for k in range(self.K):
            MC = self.envs[k]
            for _ in range(int(T * alphas[k])):
                MC.reset()
                trajectory = []
                done = False
                while not done:
                    state, done = MC.step()
                    trajectory.append(state)
                trajectories.append(trajectory)
                f[cnt] = k
                cnt += 1
        return f, trajectories

    def compute_divergence(self):
        D = math.inf
        for k, k_ in product(range(self.K), range(self.K)):
            if k == k_:
                continue
            div = 0
            for s in range(self.S):
                div += self.envs[k].pi[s] * KL(self.envs[k].P[s], self.envs[k_].P[s])
            D = min(D, div)
        return D


class MarkovChain(gym.Env):
    f"""
    Markov Chain
    S, P, mu : change via env_config
    """

    def __init__(self, H, S, P, mu):
        self.H = H  # horizon length
        self.S = S
        self.P = P
        self.mu = mu
        self.h = 0

        self.pi = self.stationary()
        # print(self.pi)
    
    def reset(self):
        self.h = 0
        self.state = np.random.choice(range(self.S), size=1, p=self.mu)[0]
        return self.state

    def step(self):
        if self.h == self.H:
            raise Exception("Exceeded horizon...")
        done = False
        if self.h == self.H - 1:
            done = True

        self.h += 1
        P = self.P[self.state]
        tmp = np.random.choice(range(self.S), size=1, p=P)
        self.state = int(tmp[0])
        return self.state, done

    def gamma_ps(self):
        P = self.P
        max_i = 20
        gamma_ps = 0
        for i in range(1, max_i + 1):
            gamma_i = spectral_gap(P, i) / i
            if gamma_i > gamma_ps:
                gamma_ps = gamma_i
        return gamma_ps

    def stationary(self):
        # Compute eigenvalues/vectors of P^T
        eigvals, eigvecs = LA.eig(self.P.T)

        # Find the eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1))
        pi = np.real(eigvecs[:, idx])

        # Normalize to sum to 1
        return pi / np.sum(pi)

if __name__ == '__main__':
    env_config = {
        'H': 100,
        'K': 3,
        'S': 5
    }
    env = MixtureMarkovChains(env_config)
    f, trajectories = env.generate_trajectories(T=10)