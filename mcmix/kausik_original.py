from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import error_rate

_MCMIX_CACHE = None


def _load_mcmix(repo_root: Path) -> Dict[str, object]:
    global _MCMIX_CACHE
    if _MCMIX_CACHE is not None:
        return _MCMIX_CACHE

    # Accept either:
    # - repo_root pointing to repository root (containing mcmix/)
    # - repo_root already pointing to the mcmix directory
    if (repo_root / "subspace.py").exists():
        mcmix_path = repo_root.resolve()
    else:
        mcmix_path = (repo_root / "mcmix").resolve()
    if not (mcmix_path / "subspace.py").exists():
        raise FileNotFoundError(f"Could not find mcmix/subspace.py at {mcmix_path}")
    if not (mcmix_path / "matplotlibrc").exists():
        raise FileNotFoundError(f"Could not find mcmix/matplotlibrc at {mcmix_path}")

    if str(mcmix_path) not in sys.path:
        sys.path.insert(0, str(mcmix_path))

    # mcmix/clustering.py calls plt.style.use('matplotlibrc') at import time.
    # Ensure relative style path resolves by importing from mcmix cwd.
    old_cwd = os.getcwd()
    os.chdir(str(mcmix_path))
    try:
        from subspace import getEig, geth  # type: ignore
        from clustering import computeStat, getClusters, getAcc  # type: ignore
        from emalg import getModelEstim, getPolicyEstim, getStartWeights, classify, em  # type: ignore
    finally:
        os.chdir(old_cwd)

    _MCMIX_CACHE = {
        "getEig": getEig,
        "geth": geth,
        "computeStat": computeStat,
        "getClusters": getClusters,
        "getAcc": getAcc,
        "getModelEstim": getModelEstim,
        "getPolicyEstim": getPolicyEstim,
        "getStartWeights": getStartWeights,
        "classify": classify,
        "em": em,
    }
    return _MCMIX_CACHE


def run_kausik_original(
    trajectories: List[List[int]],
    true_labels: np.ndarray,
    *,
    next_trajectories: List[List[int]] | None = None,
    K: int,
    S: int,
    H: int,
    mdpmix_thresh: float = 5e-6,
    mdpmix_em_iters: int = 50,
    mdpmix_em_laplace: float = 0.0,
    repo_root: Path | None = None,
    diagnostics: bool = False,
) -> Dict[str, float]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent
    fns = _load_mcmix(repo_root)

    getEig = fns["getEig"]
    geth = fns["geth"]
    computeStat = fns["computeStat"]
    getClusters = fns["getClusters"]
    getAcc = fns["getAcc"]
    getModelEstim = fns["getModelEstim"]
    getPolicyEstim = fns["getPolicyEstim"]
    getStartWeights = fns["getStartWeights"]
    classify = fns["classify"]
    em = fns["em"]

    states = np.asarray(trajectories, dtype=int)
    if next_trajectories is not None:
        nextstates = np.asarray(next_trajectories, dtype=int)
        if nextstates.shape != states.shape:
            raise ValueError(
                "next_trajectories must have the same shape as trajectories "
                f"(got states={states.shape}, nextstates={nextstates.shape})"
            )
    else:
        # Backward-compatible fallback for callers that only provide states.
        nextstates = np.concatenate([states[:, 1:], states[:, -1:]], axis=1)
    actions = np.zeros(states.shape, dtype=int)
    onehots = np.eye(S, dtype=np.float32)[states]
    onehotsp = np.eye(S, dtype=np.float32)[nextstates]
    onehotsa = onehots[..., None]  # nTraj, H, S, A(=1)
    sz = int(onehotsa.shape[0] / 3)
    omegaone = np.array([i for i in range(int(H / 4), 2 * int(H / 4))], dtype=int)
    omegatwo = np.array([i for i in range(3 * int(H / 4), H)], dtype=int)

    t0 = time.perf_counter()
    _, eigvecsa = getEig(onehotsa[:sz], onehotsp[:sz], omegaone, omegatwo, K, wt=True)
    hs = np.array(
        [
            geth(onehotsa[sz:, omegaone, :, :], onehotsp[sz:, omegaone, :]),
            geth(onehotsa[sz:, omegatwo, :, :], onehotsp[sz:, omegatwo, :]),
        ],
        dtype=np.float32,
    )
    statmns = computeStat(hs, eigvecsa, numpy=True, smalldata=True, proj=True)
    clusterlabs = getClusters(statmns, thresh=mdpmix_thresh, K=K, method="kmeans")
    mdpmix_clust_err = 1.0 - float(getAcc(clusterlabs, true_labels[sz:], K))
    clust_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    n_actions = 1
    Phat_ksa = getModelEstim(
        clusterlabs,
        states[sz:, :],
        actions[sz:, :],
        nextstates[sz:, :],
        K=K,
        nStates=S,
        nActions=n_actions,
        hard=True,
        lambda_smooth=mdpmix_em_laplace,
    )
    priorclass = np.bincount(clusterlabs, minlength=K) / len(clusterlabs)
    piclust = getPolicyEstim(
        states[sz:, :],
        actions[sz:, :],
        K,
        S,
        n_actions,
        preds=clusterlabs,
        hard=True,
        lambda_smooth=mdpmix_em_laplace,
    )
    startweights = getStartWeights(
        states[sz:, :],
        clusterlabs,
        K,
        S,
        hard=True,
        lambda_smooth=mdpmix_em_laplace,
    )
    maxapos = classify(
        Phat_ksa,
        states,
        actions,
        nextstates,
        policy=piclust,
        reg=1,
        prior=priorclass,
        startweights=startweights,
        labs=True,
        lambda_smooth=mdpmix_em_laplace,
    )
    expectclass, _, _ = em(
        maxapos,
        Phat_ksa,
        states,
        actions,
        nextstates,
        labels=true_labels,
        K=K,
        nStates=S,
        nActions=n_actions,
        prior=priorclass,
        reg=1,
        max_iter=mdpmix_em_iters,
        permute=True,
        checkin=5,
        hard=True,
        verbose=False,
        lambda_smooth=mdpmix_em_laplace,
    )
    mdpmix_em_err = float(error_rate(expectclass, true_labels))
    em_runtime = time.perf_counter() - t1

    if diagnostics:
        print(
            "[diag] mdpmix_original_clust "
            f"perm_acc={1.0 - mdpmix_clust_err:.4f} "
            f"n_pred_clusters={len(np.unique(clusterlabs))}"
        )
        print(
            "[diag] mdpmix_original_em "
            f"raw_acc={(expectclass == true_labels).mean():.4f} "
            f"perm_acc={1.0 - mdpmix_em_err:.4f} "
            f"n_pred_clusters={len(np.unique(expectclass))}"
        )

    return {
        "mdpmix_original_clust": mdpmix_clust_err,
        "mdpmix_original_em": mdpmix_em_err,
        "mdpmix_original_clust_runtime_sec": clust_runtime,
        "mdpmix_original_em_runtime_sec": em_runtime,
    }

