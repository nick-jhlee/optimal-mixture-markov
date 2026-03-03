from __future__ import annotations

import argparse
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from Clustering import InitialSpectral, LikelihoodRefinement
from mcmix.kausik_original import run_kausik_original
from utils import bootstrap_mean_ci_multi, error_rate


LASTFM_1K_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"


def _fmt_th(th: float) -> str:
    return f"{float(th):.0e}"


def _download_with_progress(url: str, output_path: Path) -> None:
    progress = tqdm(
        total=0,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {output_path.name}",
    )

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            progress.total = total_size
        downloaded = block_count * block_size
        progress.update(downloaded - progress.n)

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=_hook)
    finally:
        progress.close()


def _download_if_missing(url: str, tar_path: Path, expected_dir: Path) -> None:
    if expected_dir.exists():
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        print(f"[download] {url} -> {tar_path}")
        _download_with_progress(url, tar_path)
    print(f"[extract] {tar_path} -> {expected_dir.parent}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=expected_dir.parent)
    if not expected_dir.exists():
        raise FileNotFoundError(f"Expected extracted directory not found: {expected_dir}")


def ensure_data(data_root: Path, auto_download: bool = True) -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent
    lastfm_dir = data_root / "lastfm-dataset-1K"
    tags_dir_data = data_root / "Lastfm-ArtistTags2007"
    tags_dir_flat_data = data_root
    tags_dir_mcmix = repo_root / "mcmix" / "Lastfm-ArtistTags2007"

    if auto_download:
        _download_if_missing(
            LASTFM_1K_URL,
            data_root / "lastfm-dataset-1K.tar.gz",
            lastfm_dir,
        )
    else:
        if not lastfm_dir.exists():
            raise FileNotFoundError(f"Missing Last.fm directory: {lastfm_dir}")

    # For tags, do NOT download: use provided local copy.
    tags_candidates = [tags_dir_data, tags_dir_flat_data, tags_dir_mcmix]
    tags_dir = None
    for candidate in tags_candidates:
        if (candidate / "tags.txt").exists() and (candidate / "ArtistTags.dat").exists():
            tags_dir = candidate
            break
    if tags_dir is None:
        raise FileNotFoundError(
            "Missing local ArtistTags files required for mdpmix-style preprocessing.\n"
            "Expected both 'tags.txt' and 'ArtistTags.dat' in one of:\n"
            f" - {tags_dir_data}\n"
            f" - {tags_dir_flat_data}\n"
            f" - {tags_dir_mcmix}\n"
            "You requested local copy usage, so the script will not auto-download ArtistTags."
        )

    print(f"[data] Using Last.fm 1K directory: {lastfm_dir}")
    print(f"[data] Using ArtistTags directory: {tags_dir}")

    return lastfm_dir, tags_dir


def build_lastfm_trajectories(
    lastfm_dir: Path,
    tags_dir: Path,
    *,
    top_n_tags: int = 100,
    top_k_users: int = 10,
    trajectories_per_user: int = 75,
    horizon: int = 40,
    shuffle_seed: int | None = None,
    show_progress: bool = True,
) -> Tuple[List[List[int]], List[List[int]], np.ndarray, Dict[str, int]]:
    prep_bar = tqdm(total=6, desc="Preprocessing Last.fm", disable=not show_progress)
    genres_raw = pd.read_csv(tags_dir / "tags.txt", header=None)[0].values[:top_n_tags]
    genres = pd.DataFrame(
        [row.split()[0] for row in genres_raw],
        index=[" ".join(row.split()[1:]) for row in genres_raw],
        columns=["count"],
    )
    prep_bar.update(1)

    lastfm = pd.read_csv(
        lastfm_dir / "userid-timestamp-artid-artname-traid-traname.tsv",
        sep="\t",
        header=None,
        on_bad_lines="skip",
    )
    lastfm.columns = ["userid", "timestamp", "artistid", "artistname", "trackid", "trackname"]
    topusers = lastfm.groupby("userid").size().sort_values()[-top_k_users:]
    lastfm = lastfm[[u in topusers.index for u in lastfm["userid"]]].copy()
    lastfm.timestamp = pd.to_datetime(lastfm["timestamp"])
    prep_bar.update(1)

    tags = pd.read_table(
        tags_dir / "ArtistTags.dat",
        sep="<sep>",
        engine="python",
        header=None,
        on_bad_lines="skip",
    )
    tags.columns = ["artistid", "artistname", "tagname", "rawtagcount"]
    tags = tags[[t in genres.index for t in tags["tagname"]]]
    tags = tags[
        tags.groupby(["artistid", "artistname"])["rawtagcount"].rank(method="first", ascending=False) <= 1
    ]
    prep_bar.update(1)

    dataset = (
        lastfm.merge(tags, on="artistid")
        .sort_values(["userid", "timestamp"])[["userid", "timestamp", "tagname"]]
        .reset_index(drop=True)
    )
    tagdict = dict(zip(genres.index.values, np.arange(len(genres), dtype=int)))
    dataset["tagnum"] = dataset["tagname"].replace(tagdict).astype(int)
    prep_bar.update(1)

    # Match mdpmix notebook preprocessing: keep only transitions where genre changes.
    keeps = (dataset.tagname != dataset.tagname.shift()) * (dataset.userid == dataset.userid.shift())
    keeps.iloc[0] = True
    dataset = dataset[keeps].copy()
    prep_bar.update(1)

    users_sorted = sorted(topusers.index.values.tolist())
    user_sequences = [
        dataset[dataset.userid == user].tagnum.values
        for user in tqdm(users_sorted, desc="Collecting user sequences", disable=not show_progress)
    ]
    requested_transitions = trajectories_per_user * horizon
    # Need N+1 points per user to build N stream-consistent (s_t, s_{t+1}) pairs.
    min_available = min(len(seq) - 1 for seq in user_sequences)
    n_effective = min(requested_transitions, min_available)
    n_traj_per_user = n_effective // horizon
    n_effective = n_traj_per_user * horizon
    if n_effective <= 0:
        raise ValueError("Not enough transitions after preprocessing to form trajectories.")

    trajectories: List[List[int]] = []
    next_trajectories: List[List[int]] = []
    labels: List[int] = []
    for k, seq in tqdm(
        enumerate(user_sequences),
        total=len(user_sequences),
        desc="Building trajectories",
        disable=not show_progress,
    ):
        states = seq[:n_effective]
        nextstates = seq[1 : n_effective + 1]
        chunks = np.split(states, n_traj_per_user)
        next_chunks = np.split(nextstates, n_traj_per_user)
        trajectories.extend([chunk.astype(int).tolist() for chunk in chunks])
        next_trajectories.extend([chunk.astype(int).tolist() for chunk in next_chunks])
        labels.extend([k] * n_traj_per_user)

    # Match mdpmix notebook behavior: shuffle trajectory order only
    # (after constructing stream-consistent trajectory chunks).
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(labels))
        trajectories = [trajectories[i] for i in perm]
        next_trajectories = [next_trajectories[i] for i in perm]
        labels = [labels[i] for i in perm]
    prep_bar.update(1)
    prep_bar.close()

    meta = {
        "K": len(users_sorted),
        "S": len(genres),
        "H": horizon,
        "T": len(trajectories),
        "requested_traj_per_user": trajectories_per_user,
        "n_traj_per_user": n_traj_per_user,
        "n_effective_transitions_per_user": n_effective,
    }
    return trajectories, next_trajectories, np.asarray(labels, dtype=int), meta


def run_experiment(
    trajectories: List[List[int]],
    next_trajectories: List[List[int]],
    true_labels: np.ndarray,
    *,
    K: int,
    S: int,
    H: int,
    delta: float = 0.05,
    ours_em_iters: int = 10,
    ours_laplace: float = 0.0,
    mdpmix_threshes: Sequence[float] = (5e-6,),
    mdpmix_em_iters: int = 10,
    mdpmix_em_laplace: float = 0.0,
    show_progress: bool = True,
    diagnostics: bool = False,
) -> pd.DataFrame:
    T = len(trajectories)
    rows: List[Dict[str, float]] = []
    threshes = [float(t) for t in mdpmix_threshes]
    if len(threshes) == 0:
        raise ValueError("mdpmix_threshes must contain at least one threshold.")
    algo_bar = tqdm(total=2 + 2 * len(threshes), desc="Running algorithms", disable=not show_progress)

    t0 = time.perf_counter()
    f_hat_stage1 = InitialSpectral(trajectories, T, H, S, gamma_ps=1.0, delta=delta, K=K)
    pred_stage1 = np.asarray([f_hat_stage1[t] for t in range(T)], dtype=int)
    rows.append(
        {
            "algorithm": "ours_stage1",
            "error_rate": float(error_rate(pred_stage1, true_labels)),
            "runtime_sec": time.perf_counter() - t0,
        }
    )
    if diagnostics:
        print(
            "[diag] ours_stage1 "
            f"raw_acc={(pred_stage1 == true_labels).mean():.4f} "
            f"perm_acc={1.0 - error_rate(pred_stage1, true_labels):.4f} "
            f"n_pred_clusters={len(np.unique(pred_stage1))}"
        )
    algo_bar.update(1)

    t0 = time.perf_counter()
    f_hat_refined = LikelihoodRefinement(
        trajectories,
        f_hat_stage1,
        T,
        S,
        max_iter=ours_em_iters,
        history=False,
        lambda_smooth=ours_laplace,
    )
    pred_refined = np.asarray([f_hat_refined[t] for t in range(T)], dtype=int)
    rows.append(
        {
            "algorithm": f"ours_stage1plus2_em{ours_em_iters}",
            "error_rate": float(error_rate(pred_refined, true_labels)),
            "runtime_sec": time.perf_counter() - t0,
        }
    )
    if diagnostics:
        print(
            f"[diag] ours_stage1plus2_em{ours_em_iters} "
            f"raw_acc={(pred_refined == true_labels).mean():.4f} "
            f"perm_acc={1.0 - error_rate(pred_refined, true_labels):.4f} "
            f"n_pred_clusters={len(np.unique(pred_refined))}"
        )
    algo_bar.update(1)

    for th in threshes:
        kausik_out = run_kausik_original(
            trajectories,
            true_labels,
            next_trajectories=next_trajectories,
            K=K,
            S=S,
            H=H,
            mdpmix_thresh=th,
            mdpmix_em_iters=mdpmix_em_iters,
            mdpmix_em_laplace=mdpmix_em_laplace,
            repo_root=Path(__file__).resolve().parent,
            diagnostics=diagnostics,
        )
        th_tag = f"_th{_fmt_th(th)}"
        rows.append(
            {
                "algorithm": f"mdpmix_original_clust{th_tag}",
                "error_rate": kausik_out["mdpmix_original_clust"],
                "runtime_sec": float(kausik_out["mdpmix_original_clust_runtime_sec"]),
            }
        )
        algo_bar.update(1)
        rows.append(
            {
                "algorithm": f"mdpmix_original_em{th_tag}",
                "error_rate": kausik_out["mdpmix_original_em"],
                "runtime_sec": float(kausik_out["mdpmix_original_em_runtime_sec"]),
            }
        )
        algo_bar.update(1)
    algo_bar.close()

    return pd.DataFrame(rows)


def run_one_setting(
    *,
    h: int,
    rep: int,
    args: argparse.Namespace,
    lastfm_dir: Path,
    tags_dir: Path,
) -> pd.DataFrame:
    run_seed = args.seed + 1009 * h + rep
    np.random.seed(run_seed)
    trajectories, next_trajectories, true_labels, meta = build_lastfm_trajectories(
        lastfm_dir,
        tags_dir,
        top_n_tags=args.top_n_tags,
        top_k_users=args.top_k_users,
        trajectories_per_user=args.trajectories_per_user,
        horizon=h,
        shuffle_seed=run_seed,
        show_progress=False,
    )
    df_run = run_experiment(
        trajectories,
        next_trajectories,
        true_labels,
        K=meta["K"],
        S=meta["S"],
        H=meta["H"],
        delta=args.delta,
        ours_em_iters=args.ours_em_iters,
        ours_laplace=args.ours_laplace,
        mdpmix_threshes=args.mdpmix_thresh,
        mdpmix_em_iters=args.mdpmix_em_iters,
        mdpmix_em_laplace=args.mdpmix_em_laplace,
        show_progress=False,
        diagnostics=args.diag and rep == 0,
    )
    for k, v in meta.items():
        df_run[k] = v
    df_run["repeat"] = rep
    df_run["run_seed"] = run_seed
    return df_run


def summarize_by_h(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (h, algo), g in df.groupby(["H", "algorithm"], sort=True):
        vals = g["error_rate"].to_numpy()
        means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
        rows.append(
            {
                "H": h,
                "algorithm": algo,
                "mean": float(means[0]),
                "q025": float(lo[0]),
                "q975": float(hi[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["algorithm", "H"]).reset_index(drop=True)


def plot_error_vs_h(summary: pd.DataFrame, save_path: Path) -> None:
    th_tokens = sorted(
        {
            algo.split("_th", 1)[1]
            for algo in summary["algorithm"].unique()
            if "_th" in algo and algo.startswith("mdpmix_original_")
        },
        key=lambda s: float(s),
    )
    kausik_colors = ["#2F65A2", "#3F87C5", "#85C3E9"]

    def _style_label(algo: str) -> Tuple[str, str, str]:
        if algo.startswith("mdpmix_original_clust_th"):
            th = algo.split("_th", 1)[1]
            idx = th_tokens.index(th) % len(kausik_colors) if th in th_tokens else 0
            return kausik_colors[idx], ":", f"Kausik (th{th})"
        if algo.startswith("mdpmix_original_em_th"):
            th = algo.split("_th", 1)[1]
            idx = th_tokens.index(th) % len(kausik_colors) if th in th_tokens else 0
            return kausik_colors[idx], "-", f"Kausik (th{th}+EM)"
        if algo == "ours_stage1":
            return "#E5864A", "-", "Ours (Stage I)"
        if algo.startswith("ours_stage1plus2_em"):
            return "#D55E00", "-", "Ours (Stage I+II)"
        # Backward-compatible fallback names (single-th old outputs).
        if algo == "mdpmix_original_clust":
            return kausik_colors[1], ":", "Kausik (th5e-06)"
        if algo == "mdpmix_original_em":
            return kausik_colors[1], "-", "Kausik (th5e-06+EM)"
        return "#1f77b4", "-", algo

    plt.figure(figsize=(10, 6))
    for algo, g in summary.groupby("algorithm", sort=True):
        g = g.sort_values("H")
        color, linestyle, label = _style_label(algo)
        plt.plot(g["H"], g["mean"], marker="o", label=label, color=color, linestyle=linestyle)
        plt.fill_between(g["H"], g["q025"], g["q975"], alpha=0.15, color=color)
    plt.xlabel("H")
    plt.ylabel("Error rate")
    plt.title("Error rate vs H (Last.fm)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-world Last.fm experiment with mdpmix-style preprocessing.")
    parser.add_argument("--data-root", type=str, default="data", help="Directory for downloaded/extracted datasets.")
    parser.add_argument("--no-download", action="store_true", help="Disable automatic dataset download.")
    parser.add_argument("--top-n-tags", type=int, default=100)
    parser.add_argument("--top-k-users", type=int, default=10)
    parser.add_argument("--trajectories-per-user", type=int, default=75)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    parser.add_argument("--n-repeat", type=int, default=30)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--ours-em-iters", type=int, default=10)
    parser.add_argument(
        "--ours-laplace",
        type=float,
        default=0.0,
        help="Laplace smoothing lambda for our LikelihoodRefinement (0 disables).",
    )
    parser.add_argument(
        "--mdpmix-thresh",
        type=float,
        nargs="+",
        default=[5e-6],
        help="One or more thresholds for original mdpmix Stage-2 clustering.",
    )
    parser.add_argument("--mdpmix-em-iters", type=int, default=50)
    parser.add_argument(
        "--mdpmix-em-laplace",
        type=float,
        default=0.0,
        help="Laplace smoothing lambda for original mdpmix EM internals (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--diag", action="store_true", help="Print clustering diagnostics per run.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    lastfm_dir, tags_dir = ensure_data(data_root, auto_download=not args.no_download)
    jobs: List[Tuple[int, int]] = [(h, rep) for h in args.horizons for rep in range(args.n_repeat)]
    run_rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one_setting)(
            h=h,
            rep=rep,
            args=args,
            lastfm_dir=lastfm_dir,
            tags_dir=tags_dir,
        )
        for h, rep in jobs
    )

    df = pd.concat(run_rows, ignore_index=True)
    raw_csv = results_dir / "main_realworld_lastfm_results_raw.csv"
    df.to_csv(raw_csv, index=False)

    summary = summarize_by_h(df)
    summary_csv = results_dir / "main_realworld_lastfm_results_summary.csv"
    summary.to_csv(summary_csv, index=False)

    plot_path = results_dir / "main_realworld_error_vs_H.png"
    plot_error_vs_h(summary, plot_path)

    print("[done] Last.fm real-world comparison complete.")
    print(f"[done] Saved raw results: {raw_csv}")
    print(f"[done] Saved summary: {summary_csv}")
    print(f"[done] Saved plot: {plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
