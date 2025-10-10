from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Synthetic import MixtureMarkovChains, _build_env_config
from Clustering import InitialSpectral, LikelihoodRefinement, OracleLikelihoodRefinement
from utils import bootstrap_mean_ci_multi, error_rate
from plot_utils import plot_panels_with_ci
from matplotlib.ticker import FuncFormatter


# --------------------------- Config & utilities --------------------------- #

CB_PALETTE = {
    "known": "#D55E00",     # red for known-K (ours)
    "unknown": "#0072B2",   # blue for unknown-K variants
    "oracle": "#009E73",    # green for oracle
}


@dataclass(frozen=True)
class ExperimentConfig:
    T: int
    H: int
    K: int
    S: int
    delta: float
    em_iters: int
    # Sweep of (c1, c2) for unknown-K: factors for sigma and rho thresholds
    c_pairs: Tuple[Tuple[float, float], ...]
    seed: int | None = None


# --------------------------- Per-repeat run --------------------------- #

def run_one_repeat(cfg: ExperimentConfig, repeat_idx: int) -> Dict:
    """
    For a fixed (T, H), produce error for:
    - Known-K: Stage I (K known) + EM refinement (cfg.em_iters)
    - Unknown-K: Stage I (K unknown) + EM refinement, sweeping (c1, c2)

    Returns dict with scalar errors per algorithm variant.
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed + 37 * repeat_idx + 1009 * cfg.T + 17 * cfg.H)

    env_config = _build_env_config(cfg.H, cfg.K, cfg.S)
    env = MixtureMarkovChains(env_config)
    gamma_ps = env.gamma_ps
    f_true, trajectories = env.generate_trajectories(cfg.T)
    true_arr = np.fromiter((f_true[t] for t in range(cfg.T)), dtype=int)

    # Oracle baseline
    f_oracle = OracleLikelihoodRefinement(trajectories, env)
    pred_oracle = np.fromiter((f_oracle[t] for t in range(cfg.T)), dtype=int)
    err_oracle = error_rate(pred_oracle, true_arr)

    # Known-K pipeline: Stage I and Stage I+II (EM10)
    f_hat_known = InitialSpectral(
        trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=cfg.K
    )
    pred_known_s1 = np.fromiter((f_hat_known[t] for t in range(cfg.T)), dtype=int)
    err_known_s1 = error_rate(pred_known_s1, true_arr)

    f_known_ref = LikelihoodRefinement(
        trajectories, f_hat_known, cfg.T, cfg.S, max_iter=cfg.em_iters, history=False
    )
    pred_known_em = np.fromiter((f_known_ref[t] for t in range(cfg.T)), dtype=int)
    err_known_em = error_rate(pred_known_em, true_arr)

    # Unknown-K pipeline under different (c1, c2)
    unknown_errors: Dict[str, float] = {}
    for (c1, c2) in cfg.c_pairs:
        # Normalize float keys to scientific notation (e.g., 1e-04), for consistent legend labels
        c1s = format(c1, '.0e')
        c2s = format(c2, '.0e')

        # Stage I (unknown K)
        f_hat_unknown = InitialSpectral(
            trajectories, cfg.T, cfg.H, cfg.S, gamma_ps, cfg.delta, K=None, c1=c1, c2=c2
        )
        pred_unknown_s1 = np.fromiter((f_hat_unknown[t] for t in range(cfg.T)), dtype=int)
        unknown_errors[f"unknown_c1{c1s}_c2{c2s}_stage1"] = error_rate(pred_unknown_s1, true_arr)

        # Stage I+II (EM10) for unknown K
        f_unknown_ref = LikelihoodRefinement(
            trajectories, f_hat_unknown, cfg.T, cfg.S, max_iter=cfg.em_iters, history=False
        )
        pred_unknown_em = np.fromiter((f_unknown_ref[t] for t in range(cfg.T)), dtype=int)
        unknown_errors[f"unknown_c1{c1s}_c2{c2s}_em{cfg.em_iters}"] = error_rate(pred_unknown_em, true_arr)

    out = {
        "T": cfg.T,
        "H": cfg.H,
        "repeat": repeat_idx,
        "oracle": err_oracle,
        "known_stage1": err_known_s1,
        f"known_em{cfg.em_iters}": err_known_em,
    }
    out.update(unknown_errors)
    return out


# --------------------------- Orchestration --------------------------- #

def run_grid(
    T_list: Sequence[int],
    H_list: Sequence[int],
    *,
    K: int = 3,
    S: int = 10,
    delta: float = 0.05,
    em_iters: int = 10,
    c_pairs: Sequence[Tuple[float, float]] = ((1e-4, 1e-1), (5e-5, 5e-2), (2e-4, 2e-1)),
    n_repeat: int = 20,
    n_jobs: int = -1,
    base_seed: int | None = 2025,
) -> pd.DataFrame:
    """Run all (T,H) × repeats and return a tidy DataFrame of per-repeat errors."""
    jobs: List[ExperimentConfig] = []
    for T in T_list:
        for H in H_list:
            for r in range(n_repeat):
                jobs.append(
                    ExperimentConfig(
                        T=T, H=H, K=K, S=S, delta=delta,
                        em_iters=em_iters,
                        c_pairs=tuple(c_pairs),
                        seed=base_seed,
                    )
                )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_repeat)(cfg, r) for r, cfg in enumerate(jobs)
    )
    df = pd.DataFrame(results).sort_values(["T", "H", "repeat"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Bootstrap mean & CIs for each (T,H,algorithm). Returns tidy DataFrame:
    columns = [T, H, algo, mean, lo, hi]
    """
    algos = [c for c in df.columns if c not in ("T", "H", "repeat")]
    rows = []
    for (T, H), g in df.groupby(["T", "H"], sort=True):
        for algo in algos:
            vals = g[algo].to_numpy()
            means, lo, hi = bootstrap_mean_ci_multi(vals[:, None], alpha=alpha)
            rows.append(
                {"T": T, "H": H, "algo": algo,
                 "mean": float(means[0]),
                 "lo": float(lo[0]),
                 "hi": float(hi[0])}
            )
    out = pd.DataFrame(rows).sort_values(["T", "H", "algo"]).reset_index(drop=True)
    return out


# --------------------------- Plotting --------------------------- #

def _style_for(algo: str) -> Tuple[str, str]:
    # Fallback global style (unused by plot_panels which builds a local style with colors per (c1,c2))
    if algo == "oracle":
        return CB_PALETTE["oracle"], ":"
    if algo == "known_stage1":
        return CB_PALETTE["known"], "--"
    if algo.startswith("known_em"):
        return CB_PALETTE["known"], "-"
    if algo.startswith("unknown_c1"):
        return CB_PALETTE["unknown"], (0, (2, 2)) if algo.endswith("stage1") else "-"
    return "#1F77B4", (0, (2, 2))


def plot_panels(summary: pd.DataFrame, savepath: str = "results/ablation2_results.pdf", h_split: Tuple[Sequence[int], Sequence[int]] | None = None) -> None:
    # Series and labels: show unknown-K Stage I (dotted) and Stage I+II (solid) per (c1,c2)
    all_algos = [c for c in summary["algo"].unique()]
    unknown_s1 = sorted([a for a in all_algos if a.startswith("unknown_c1") and a.endswith("stage1")])
    unknown_em = sorted([a for a in all_algos if a.startswith("unknown_c1") and ("_em" in a) and not a.endswith("stage1")])

    # Parse c1 and c2 from algo into new columns to facet by c1 and label by c2
    def parse_c1_c2(algo_name: str) -> Tuple[str, str]:
        base = algo_name.split("_stage1")[0].split("_em")[0]
        # base format: unknown_c1{val}_c2{val}
        _, c1s, c2s = base.split("_")
        c1v = c1s.replace("c1", "")
        c2v = c2s.replace("c2", "")
        return c1v, c2v

    # Filter summary to unknown-only series for plotting (avoid early use of series_order)
    summary_plot = summary[summary["algo"].str.startswith("unknown_c1")].copy()
    c1_vals: List[str] = []
    c2_vals: List[str] = []
    stages: List[str] = []
    for a in summary_plot["algo"].tolist():
        c1v, c2v = parse_c1_c2(a)
        c1_vals.append(c1v)
        c2_vals.append(c2v)
        stages.append("stage1" if a.endswith("stage1") else "em")
    summary_plot["c1"] = c1_vals
    summary_plot["c2"] = c2_vals
    summary_plot["stage"] = stages
    # Enforce numeric ascending order for c1 facets
    try:
        summary_plot["c1"] = summary_plot["c1"].astype(float)
    except Exception:
        pass

    # Helper function for formatting scientific notation
    def _format_sci(val_str: str) -> str:
        try:
            v = float(val_str)
            s = format(v, ".0e")  # e.g., '1e-02', '1e+00'
            # strip leading '+' and leading zeros in exponent
            if "e" in s:
                base, exp = s.split("e")
                exp = exp.lstrip("+")
                if exp.startswith("-"):
                    exp = "-" + exp[1:].lstrip("0")
                    if exp == "-":
                        exp = "0"
                else:
                    exp = exp.lstrip("0") or "0"
                return f"{base}e{exp}"
            return s
        except Exception:
            return val_str

    # Add H_group if h_split is provided
    if h_split is not None:
        h_list_1, h_list_2 = h_split
        
        # Find H values that appear in both lists (need to duplicate these rows)
        h_overlap = set(h_list_1) & set(h_list_2)
        
        # Create separate dataframes for each group
        dfs_to_concat = []
        
        # Group 1: all H values in h_list_1
        df_group1 = summary_plot[summary_plot["H"].isin(h_list_1)].copy()
        df_group1["H_group"] = 1
        dfs_to_concat.append(df_group1)
        
        # Group 2: all H values in h_list_2
        df_group2 = summary_plot[summary_plot["H"].isin(h_list_2)].copy()
        df_group2["H_group"] = 2
        dfs_to_concat.append(df_group2)
        
        # Concatenate both groups (this duplicates H values that are in both lists)
        summary_plot = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Get sorted c1 values
        c1_sorted = sorted(summary_plot["c1"].unique())
        
        # Create composite facet key with explicit ordering (row-major)
        # Format: "1_c1=0.001" so it sorts H_group first, then c1
        summary_plot["facet_key"] = summary_plot.apply(
            lambda r: f"{int(r['H_group'])}_c1={r['c1']}",
            axis=1
        )
        
        # Create readable facet labels for display (just show c1 value)
        summary_plot["facet_label"] = summary_plot.apply(
            lambda r: f"c1={_format_sci(str(r['c1']))}",
            axis=1
        )
        
        facet_col_to_use = "facet_key"
    else:
        facet_col_to_use = "c1"

    # Create a normalized series key so that series are consistent across facets (by c2 and stage only)
    summary_plot["series_key"] = summary_plot.apply(
        lambda r: f"c2={r['c2']}|{('Stage I' if r['stage']== 'stage1' else 'Stage I+II')}",
        axis=1,
    )

    # Determine series order and legend labels (vertical alignment: Stage I then Stage II by c2)
    c2_unique = sorted(summary_plot["c2"].unique(), key=lambda s: float(s))
    # Interleave Stage I and Stage I+II for each c2 so legend fills correctly in row-major order
    series_order = []
    for c2v in c2_unique:
        series_order.append(f"c2={c2v}|Stage I")
        series_order.append(f"c2={c2v}|Stage I+II")

    # Legend labels with normalized scientific c2 (e.g., 1e-2, 1e-1, 1e0)
    label_map: Dict[str, str] = {}
    for key in series_order:
        c2_part, stage_part = key.split("|")
        c2_raw = c2_part.replace("c2=", "")
        c2_lbl = _format_sci(c2_raw)
        label_map[key] = f"c2={c2_lbl} ({stage_part})"

    # Palette and colors per c2 (lighter->darker for smaller->larger c2)
    reds_palette = ["#F46D43", "#D73027", "#A50026"]  # Coral, Rust red, Deep maroon
    if len(c2_unique) == 1:
        c2_colors = {c2_unique[0]: reds_palette[0]}
    else:
        # Map c2 positions to indices in reds_palette, ensuring darkest for largest c2
        c2_colors = {}
        n_levels = len(c2_unique)
        for i, c2v in enumerate(c2_unique):
            idx = int(round(i * (len(reds_palette) - 1) / (n_levels - 1)))
            c2_colors[c2v] = reds_palette[idx]

    # Local style function capturing color by c2 and stage styles
    def style_for_local(series_key: str) -> Tuple[str, str]:
        # series_key: "c2={val}|Stage I" or "c2={val}|Stage I+II"
        c2_part, stage_part = series_key.split("|")
        c2v = c2_part.replace("c2=", "")
        color = c2_colors.get(c2v, CB_PALETTE["unknown"])  # fallback
        if stage_part == "Stage I":
            return color, (0, (4, 2))
        return color, "-"

    # Use c1 as facet column; build legend order by label (row-major): Stage I row, then Stage II row
    # Column-major legend order for vertical alignment per c2: (Stage I, Stage I+II) per column
    legend_order_labels: List[str] = []
    for c2v in c2_unique:
        c2_lbl = _format_sci(c2v)
        legend_order_labels.append(f"c2={c2_lbl} (Stage I)")
        legend_order_labels.append(f"c2={c2_lbl} (Stage I+II)")

    # Build facet title map and x-tick formatter per facet if using h_split
    facet_title_map = None
    x_tick_formatter_per_facet = None
    if h_split is not None:
        facet_title_map = {}
        x_tick_formatter_per_facet = {}
        for facet_key in summary_plot["facet_key"].unique():
            # Get corresponding label and H_group
            matching_rows = summary_plot[summary_plot["facet_key"] == facet_key]
            if not matching_rows.empty:
                facet_title_map[facet_key] = matching_rows.iloc[0]["facet_label"]
                h_group = matching_rows.iloc[0]["H_group"]
                # Use scientific notation (1e3, 2e3, etc.) for H group 2 (bottom row)
                if h_group == 2:
                    x_tick_formatter_per_facet[facet_key] = lambda x: f"{int(x/1000)}e3"
                else:
                    x_tick_formatter_per_facet[facet_key] = lambda x: f"{int(x)}"

    plot_panels_with_ci(
        summary_plot,
        series_order=series_order,
        label_map=label_map,
        style_for=style_for_local,
        legend_order=series_order,
        legend_ncol=len(c2_unique),  # num columns = num c2 values (each col has Stage I on top, Stage I+II on bottom)
        savepath=savepath,
        facet_col=facet_col_to_use,
        x_col="H",
        facet_title_map=facet_title_map,
        x_tick_formatter_per_facet=x_tick_formatter_per_facet,
        series_col="series_key",
        y_mean="mean",
        y_lo="lo",
        y_hi="hi",
        inset=False,
        # With three H values like [10000,20000,30000], ensure the middle tick appears
        inset_xlim=(10, 100),
        inset_xticks=list(range(100, 1001, 100)),
        extra_xticks=None,
        y_label="Clustering error",
        title_fontsize=18,
        label_fontsize=16,
        tick_fontsize=14,
        legend_fontsize=16,
        legend_bbox=(0.5, 1.10),
        top_adjust=0.88,
    )


# --------------------------- CLI entry --------------------------- #

def main():
    T_list = [100]
    # T_list = [100, 200, 300, 400, 500, 600]
    
    # Two H ranges for two-row layout (1000 appears in both)
    H_list_1 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    H_list_2 = [1000, 90000]
    # H_list_2 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
    # Include all unique H values
    H_list = sorted(set(H_list_1 + H_list_2))

    # Define grids for c1 and c2 each - using 3 c1 values for 3 columns
    c1_values = [1e-3, 1e-2, 1e-1]
    # c2_values = [1e-2, 1e-1, 1]
    c2_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    c_pairs = tuple((c1, c2) for c1 in c1_values for c2 in c2_values)

    df = run_grid(
        T_list=T_list,
        H_list=H_list,
        K=3,
        S=10,
        delta=0.05,
        em_iters=10,
        c_pairs=c_pairs,
        n_repeat=30,
        n_jobs=-1,
        base_seed=2025,
    )
    df.to_csv("results/ablation2_results_unknownK_raw.csv", index=False)

    summary = summarize(df, alpha=0.05)
    summary.to_csv("results/ablation2_results_unknownK_summary.csv", index=False)

    # Plot with two-row layout (H_list_1 in row 1, H_list_2 in row 2)
    plot_panels(summary, savepath="results/ablation2_results.png", h_split=(H_list_1, H_list_2))
    plot_panels(summary, savepath="results/ablation2_results.pdf", h_split=(H_list_1, H_list_2))


if __name__ == "__main__":
    main()
