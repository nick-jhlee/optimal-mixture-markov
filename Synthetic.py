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

# def _build_env_config(H: int, K: int, S: int) -> Dict:
#     """Match your special K=2 construction; otherwise let Synthetic randomize."""

#     ## Construction considered in the paper (Appendix )
#     if K == 2:
#         if S % 2:
#             S += 1
#         # Vectorized construction of P1, P2 as SxS row-stochastic matrices
#         half = S // 2
#         idx = np.arange(S)
#         col_mask = (idx[None, :] >= half).astype(float)  # shape (1, S), indicates s_ >= S//2
#         right = np.tile(col_mask, (S, 1))                # broadcast across rows -> (S, S)
#         left = 1.0 - right

#         P1 = right * 2.1 + left * 1.9
#         P2 = right * 1.9 + left * 2.1
#         P1 /= P1.sum(axis=1, keepdims=True)
#         P2 /= P2.sum(axis=1, keepdims=True)
#         return {"H": H, "K": K, "S": S, "Ps": [P1, P2],
#                 "mus": [np.ones(S) / S, np.ones(S) / S]}
#     return {"H": H, "K": K, "S": S}

# def _build_env_config(H: int, K: int, S: int) -> Dict:
#     """MMC instance with D_pi >> Delta_W^2 via a doubly-stochastic construction.
#        Chain 1: P1 has uniform rows (rank-1).
#        Chain 2: P2 shifts epsilon mass in each row from (s+1) to s (mod S).
#        Both chains share uniform stationary distribution pi = 1/S.
#     """
#     if K == 2:
#         # choose epsilon < 1/S to keep strict positivity
#         eps = 0.5 / S  # small, safe; feel free to tweak (e.g., 0.1/S)
#         u = np.full(S, 1.0 / S, dtype=float)

#         # P1: every row uniform
#         P1 = np.tile(u, (S, 1))

#         # P2: for each row s, bump column s by +eps, column (s+1)%S by -eps
#         P2 = np.tile(u, (S, 1))
#         for s in range(S):
#             P2[s, s] += eps
#             P2[s, (s + 1) % S] -= eps

#         # numerical safety (should already be >0)
#         if np.any(P2 <= 0):
#             raise ValueError("Choose smaller eps; P2 has nonpositive entries.")
#         if not np.allclose(P2.sum(axis=1), 1.0):
#             raise AssertionError("Rows not normalized.")
#         if not np.allclose(P2.sum(axis=0), 1.0, atol=1e-12):
#             # Should be doubly-stochastic, but tolerances vary; adjust if needed.
#             pass

#         mus = [u.copy(), u.copy()]  # start from the common stationary distribution

#         return {"H": H, "K": K, "S": S, "Ps": [P1, P2], "mus": mus}

#     # Fallback for other K
#     return {"H": H, "K": K, "S": S}


def _build_env_config(H: int, K: int, S: int) -> Dict:
    """
    MMC instance with D_pi >> Delta_W^2 via a doubly-stochastic 'cyclic bump'.

    - All chains share uniform stationary distribution pi = 1/S.
    - For each chain k, every row is uniform except for a tiny cyclic bump:
        add +eps to column (s + shift_k) and -eps to column (s + shift_k + 1) (mod S).
      This preserves row-stochasticity and positivity for eps < 1/S.
    - Shifts are spaced around the ring; when K > S, shifts repeat (cycling).

    Returns the dict expected by MixtureMarkovChains: {"H","K","S","Ps","mus"}.
    """
    assert S >= 2, "Need at least two states."
    if K <= 0:
        raise ValueError("K must be >= 1")

    # small enough to keep all entries positive: 1/S - eps > 0
    eps = 0.9 / S
    u = np.full(S, 1.0 / S, dtype=float)          # uniform stationary distribution
    mus = [u.copy() for _ in range(K)]

    U = np.tile(u, (S, 1))

    # Evenly spaced shifts around the ring.
    if K <= S:
        # shifts = np.floor(np.linspace(0, S - 1, K - 1, endpoint=False)).astype(int)
        shifts = np.arange(K - 1)
    else:
        raise ValueError("K must be <= S")

    Ps = [U]
    for sh in shifts:
        P = U.copy()                    # start from uniform rows
        # Apply the bump per row with the chosen shift
        for s in range(S):
            j_plus  = (s + sh) % S
            j_minus = (s + sh + 1) % S            # NOTE the +1 shift (cyclic)
            P[s, j_plus]  += eps
            P[s, j_minus] -= eps

        # safety checks
        if np.any(P <= 0):
            raise ValueError(
                f"Nonpositive entry in P (eps too large). Try eps < 1/S; got eps={eps}, S={S}."
            )
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("Row sums not equal to 1 after bump.")

        Ps.append(P)

    return {"H": H, "K": K, "S": S, "Ps": Ps, "mus": mus}

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
        self.D_pi, self.Delta_W = self.compute_divergence_and_Delta_W()
        
    def generate_trajectories(self, T, alphas=None):
        """
        Generate exactly T trajectories of length H with prescribed cluster sizes.
        Default cluster sizes are uniform across K clusters.
        """
        if alphas is None:
            alphas = [1/self.K for _ in range(self.K)]
        if len(alphas) != self.K:
            raise Warning("Number of alphas must be equal to the number of Markov Chain Models... Regenerating the alphas with uniform distribution")
            alphas = [1/self.K for _ in range(self.K)]

        # Use multinomial draw to ensure counts sum to T
        probs = np.array(alphas, dtype=float)
        probs = probs / probs.sum()
        counts = np.random.multinomial(T, probs)

        trajectories = []
        f = {}
        cnt = 0
        for k in range(self.K):
            MC = self.envs[k]
            for _ in range(counts[k]):
                MC.reset()
                trajectory = []
                done = False
                while not done:
                    state, done = MC.step()
                    trajectory.append(state)
                trajectories.append(trajectory)
                f[cnt] = k
                cnt += 1
        # Safety check: ensure T trajectories generated
        assert len(trajectories) == T and cnt == T
        return f, trajectories

    def compute_divergence_and_Delta_W(self):
        D_pi, Delta_W = math.inf, math.inf
        for k, k_ in product(range(self.K), range(self.K)):
            if k == k_:
                continue
            pi_k, pi_k_ = self.envs[k].pi, self.envs[k_].pi
            P_k, P_k_ = self.envs[k].P, self.envs[k_].P

            D_pi_k, Delta_W_k = 0.0, 0.0
            for s in range(self.S):
                D_pi_k += pi_k[s] * KL(P_k[s], P_k_[s])
                sqrt_pi_k_s, sqrt_pi_k_s_ = np.sqrt(pi_k[s]), np.sqrt(pi_k_[s])
                Pk_s = P_k[s]
                Pk_s_ = P_k_[s]
                for s_ in range(self.S):
                    Delta_W_k += (sqrt_pi_k_s * Pk_s[s_] - sqrt_pi_k_s_ * Pk_s_[s_])**2
            D_pi = min(D_pi, D_pi_k)
            Delta_W = min(Delta_W, Delta_W_k)

        return D_pi, Delta_W

    def compute_divergence(self):
        return self.D_pi

    def compute_Delta_W(self):
        return self.Delta_W


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