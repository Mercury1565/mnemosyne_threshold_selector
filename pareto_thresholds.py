"""
Pareto-frontier threshold selection for the 3-mode reuse pipeline (copy / IA reuse / fresh).
Sweeps every (t_l, t_h) pair, keeps the non-dominated frontier, applies RULES to pick the
fastest pair that's still safe enough. Runs once per combo in COMBOS.

Usage:  python pareto_thresholds.py
        python pareto_thresholds.py --max-ia-risk 0.2 --max-copy-risk 0.1 --min-speedup 5
"""
import os
import itertools
import argparse
import subprocess
import numpy as np, pandas as pd
from scipy.stats import beta
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SHEET_SYNC_URL = "http://localhost:5678/webhook/mnemosyne/csv"
DOC_ID = "1UbNfo8payXR623W4ymE0Exm8tiffItqCKF0xU1_1cjI"

def sync_csv_to_sheet(csv_path, sheet_name=None, sheet_name_prefix=None):
    """Push a written CSV to the Google Sheet webhook (sheet name = file's basename)."""
    if not sheet_name:
        sheet_name = os.path.splitext(os.path.basename(csv_path))[0]

    if sheet_name_prefix:
        sheet_name = sheet_name_prefix + sheet_name

    try:
        subprocess.run(
            ["curl", "-X", "POST", "-F", f"docId={DOC_ID}", "-F", f"sheetName={sheet_name}", "-F", f"csv=@{csv_path}", SHEET_SYNC_URL],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"! sheet sync failed for {csv_path}: {e}")

# CONFIG
CONFIG = dict(
    results = "./data/tier3_all_models_per_frame_no_tier_name.csv",  # IA-reuse run, all 5 combos (baseline_ms, optimized_ms, f1_gap, miou_gap)
    deltas  = "./data/consecutive_frame_metric_deltas.csv",    # frame-to-frame F1/mIoU, source of loss_copy
    oracle  = "./data/inference_oracle_mc.csv",                # similarity score per frame
    score_col = "S_oracle_ego_obj",                            # oracle column used as the routing score

    TAU    = 0.1,                # per-frame loss tolerance (F1 / mIoU units) above which a frame counts as harmed
    CONF   = 0.90,               # confidence level of the Clopper-Pearson risk bound
    CHECKER_MS = 5.0,            # TODO: measure
    MOTION_COMP_MS = 5.0,        # TODO: measure
    MIN_SCORE_THRESHOLD = 0.25,  # hard floor: t_l never goes below this

    RISK_TOLERANCE_GRID = np.round(np.arange(0.10, 1.001, 0.01), 3),  # shared risk caps swept in constrained_thresholds

    out_dir = "./outputs",                                          # where every plot/CSV below gets written
    curves_plot_png = "pareto_threshold_curves.png",                # 4-panel risk/speedup/delta curves vs threshold
    tolerance_plot_png = "pareto_thresholds_vs_tolerance.png",      # t_l/t_h vs shared risk tolerance cap
    rules_sweep_csv = "rules_sweep.csv",                            # RULES_SWEEP permutation results
    save_joined_table = True,                                       # write the full joined tab (see load()) per combo
    joined_table_csv = "joined_table.csv",                          # so any row's n_ia/n_copy can be traced by hand
)

COMBOS = [
    dict(model="pointpillars", oracle_dataset="nuscenes", results_dataset="nuscenes",  results_stride=4),
    dict(model="centerpoint",  oracle_dataset="nuscenes", results_dataset="nuscenes",  results_stride=4),
    dict(model="3dssd",        oracle_dataset="kitti",    results_dataset="kitti_raw", results_stride=1),
    dict(model="pvrcnn",       oracle_dataset="kitti",    results_dataset="kitti_raw", results_stride=1),
    dict(model="m3detr",       oracle_dataset="waymo",    results_dataset="waymo",     results_stride=1),
]

# RULES_SWEEP -- the grid of RULES values to permute for rules_sweep.csv.
RULES_SWEEP = dict(
    max_ia_risk_grid   = np.round(np.arange(0.10, 1.01, 0.05), 3),
    max_copy_risk_grid = np.round(np.arange(0.10, 1.01, 0.05), 3),
    min_speedup_grid   = [None, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
)

# RULES -- the algorithmic selection criteria.
RULES = dict(
    max_ia_risk = 0.30,     # reject any pair whose IA-region risk bound exceeds this
    max_copy_risk = 0.30,   # reject any pair whose copy-region risk bound exceeds this
    min_speedup = None,     # optional floor; None = no floor
)

def rules_with_overrides(rules):
    """Override RULES with any provided --flags."""
    parser = argparse.ArgumentParser(description="Override RULES for this run.")
    parser.add_argument("--max-ia-risk", type=float, default=None)
    parser.add_argument("--max-copy-risk", type=float, default=None)
    parser.add_argument("--min-speedup", type=float, default=None)
    args = parser.parse_args()

    overridden = dict(rules)
    for key, value in vars(args).items():
        if value is not None:
            overridden[key] = value
    return overridden

def scene_key(scene_series, oracle_dataset):
    """Canonical join key per scene. KITTI's files spell scenes differently
    ('0001' vs '1' vs '..._drive_0001_sync'); no-op for nuScenes/Waymo."""
    if oracle_dataset == "kitti":
        scene_str = scene_series.astype(str)
        drive_number = scene_str.str.extract(r"drive_(\d+)")[0]
        return drive_number.fillna(scene_str).astype(int)
    return scene_series

# LOAD & JOIN
def load(cfg):
    results_df = pd.read_csv(cfg["results"])
    deltas_df = pd.read_csv(cfg["deltas"])
    oracle_df = pd.read_csv(cfg["oracle"])

    deltas_df = deltas_df[deltas_df.model == cfg["model"]].copy()
    oracle_df = oracle_df[(oracle_df.model == cfg["model"]) & (oracle_df.dataset == cfg["oracle_dataset"])].copy()
    results_df = results_df[(results_df.model == cfg["model"]) &
                             (results_df.dataset == cfg["results_dataset"])].copy()
    results_df = results_df.rename(columns={"baseline_ms": "fresh_ms", "optimized_ms": "reuse_ms"})

    deltas_df["scene_key"] = scene_key(deltas_df.scene, cfg["oracle_dataset"])
    oracle_df["scene_key"] = scene_key(oracle_df.scene, cfg["oracle_dataset"])
    results_df["scene_key"] = scene_key(results_df.scene, cfg["oracle_dataset"])

    joined = deltas_df.merge(oracle_df[["scene_key", "frame_cur", cfg["score_col"]]],
                left_on=["scene_key", "current_dataset_frame"], right_on=["scene_key", "frame_cur"], how="inner")
    joined = joined.rename(columns={cfg["score_col"]: "score"})

    f1_drop = joined.previous_f1 - joined.current_f1
    miou_drop = joined.previous_miou - joined.current_miou
    joined["loss_copy"] = np.maximum(f1_drop, miou_drop)

    results_df = results_df[results_df.reset == 0].copy()
    results_df["current_frame"] = results_df.frame * cfg["results_stride"]
    results_df["loss_ia"] = np.maximum(results_df.f1_gap, results_df.miou_gap)

    joined = joined.merge(results_df[["scene_key", "current_frame", "loss_ia", "f1_gap", "miou_gap",
                                       "fresh_ms", "reuse_ms", "speedup"]],
                on=["scene_key", "current_frame"], how="left")

    return joined

def SCORE_GRID(tab, min_threshold):
    """Candidate score thresholds: data's own min score, floored to min_threshold --
    below that, fresh inference is mandatory no matter what the risk bound says."""
    data_floor = np.floor(tab.score.min() * 100) / 100
    floor = max(data_floor, min_threshold)
    return np.round(np.arange(floor, 1.001, 0.01), 3)

# HELPERS
def cp_upper(num_successes, num_trials, conf):
    """Clopper-Pearson upper bound on a binomial proportion."""
    if num_trials == 0 or num_successes >= num_trials:
        return 1.0
    return beta.ppf(conf, num_successes + 1, num_trials - num_successes)

# PARETO SWEEP
def sweep(tab, cfg):
    """Evaluate every (t_l, t_h) pair on GRID with t_l <= t_h."""
    ia_tab = tab.dropna(subset=["loss_ia"])
    fresh_ms = ia_tab.fresh_ms.median()
    reuse_ms = ia_tab.reuse_ms.median()
    copy_ms  = cfg["CHECKER_MS"] + cfg["MOTION_COMP_MS"]

    score, loss_copy = tab.score.values, tab.loss_copy.values
    ia_score, ia_loss = ia_tab.score.values, ia_tab.loss_ia.values

    rows = []
    for lower_thr, upper_thr in itertools.combinations_with_replacement(cfg["GRID"], 2):
        frac_fresh = (score < lower_thr).mean()
        frac_ia    = ((score >= lower_thr) & (score < upper_thr)).mean()
        frac_copy  = (score >= upper_thr).mean()

        # time-weighted, not an average of per-frame ratios
        speedup = fresh_ms / (frac_fresh * fresh_ms + frac_ia * reuse_ms + frac_copy * copy_ms)

        ia_mask = (ia_score >= lower_thr) & (ia_score < upper_thr)
        ia_risk = cp_upper(int((ia_loss[ia_mask] > cfg["TAU"]).sum()), int(ia_mask.sum()), cfg["CONF"])

        copy_mask = score >= upper_thr
        copy_risk = cp_upper(int((loss_copy[copy_mask] > cfg["TAU"]).sum()), int(copy_mask.sum()), cfg["CONF"])

        rows.append((lower_thr, upper_thr, speedup, ia_risk, copy_risk))

    return pd.DataFrame(rows, columns=["t_l", "t_h", "speedup", "ia_risk", "copy_risk"])

def threshold_curve_by_t_h(tab, cfg, fixed_t_l):
    """Sweep t_h over GRID (t_l held fixed). Copy-region metrics as t_h moves."""
    ia_tab = tab.dropna(subset=["loss_ia"])
    fresh_ms = ia_tab.fresh_ms.median()
    reuse_ms = ia_tab.reuse_ms.median()
    copy_ms  = cfg["CHECKER_MS"] + cfg["MOTION_COMP_MS"]

    score, loss_copy = tab.score.values, tab.loss_copy.values
    f1_drop = (tab.previous_f1 - tab.current_f1).values
    miou_drop = (tab.previous_miou - tab.current_miou).values

    rows = []
    for t_h in cfg["GRID"][cfg["GRID"] >= fixed_t_l]:
        frac_fresh = (score < fixed_t_l).mean()
        frac_ia    = ((score >= fixed_t_l) & (score < t_h)).mean()
        frac_copy  = (score >= t_h).mean()
        speedup = fresh_ms / (frac_fresh * fresh_ms + frac_ia * reuse_ms + frac_copy * copy_ms)

        copy_mask = score >= t_h
        num_copy = int(copy_mask.sum())
        risk_cap = cp_upper(int((loss_copy[copy_mask] > cfg["TAU"]).sum()), num_copy, cfg["CONF"])
        delta_miou = miou_drop[copy_mask].mean() if num_copy else np.nan
        delta_accuracy = f1_drop[copy_mask].mean() if num_copy else np.nan

        rows.append((t_h, risk_cap, speedup, delta_miou, delta_accuracy))

    return pd.DataFrame(rows, columns=["threshold", "risk_cap", "speedup", "delta_miou", "delta_accuracy"])

def threshold_curve_by_t_l(tab, cfg, fixed_t_h):
    """Sweep t_l over GRID (t_h held fixed). IA-region metrics as t_l moves."""
    ia_tab = tab.dropna(subset=["loss_ia"])
    fresh_ms = ia_tab.fresh_ms.median()
    reuse_ms = ia_tab.reuse_ms.median()
    copy_ms  = cfg["CHECKER_MS"] + cfg["MOTION_COMP_MS"]

    score = tab.score.values
    ia_score, ia_loss = ia_tab.score.values, ia_tab.loss_ia.values
    ia_f1_gap, ia_miou_gap = ia_tab.f1_gap.values, ia_tab.miou_gap.values

    rows = []
    for t_l in cfg["GRID"][cfg["GRID"] <= fixed_t_h]:
        frac_fresh = (score < t_l).mean()
        frac_ia    = ((score >= t_l) & (score < fixed_t_h)).mean()
        frac_copy  = (score >= fixed_t_h).mean()
        speedup = fresh_ms / (frac_fresh * fresh_ms + frac_ia * reuse_ms + frac_copy * copy_ms)

        ia_mask = (ia_score >= t_l) & (ia_score < fixed_t_h)
        num_ia = int(ia_mask.sum())
        risk_cap = cp_upper(int((ia_loss[ia_mask] > cfg["TAU"]).sum()), num_ia, cfg["CONF"])
        delta_miou = ia_miou_gap[ia_mask].mean() if num_ia else np.nan
        delta_accuracy = ia_f1_gap[ia_mask].mean() if num_ia else np.nan

        rows.append((t_l, risk_cap, speedup, delta_miou, delta_accuracy))

    return pd.DataFrame(rows, columns=["threshold", "risk_cap", "speedup", "delta_miou", "delta_accuracy"])

def pareto_front(sweep_df):
    """Non-dominated subset: speedup high is good, ia_risk/copy_risk low is good."""
    speedup, ia_risk, copy_risk = sweep_df.speedup.values, sweep_df.ia_risk.values, sweep_df.copy_risk.values

    at_least_as_good = ((speedup[:, None] >= speedup[None, :]) &
                         (ia_risk[:, None]  <= ia_risk[None, :]) &
                         (copy_risk[:, None] <= copy_risk[None, :]))
    strictly_better  = ((speedup[:, None] > speedup[None, :]) |
                         (ia_risk[:, None]  < ia_risk[None, :]) |
                         (copy_risk[:, None] < copy_risk[None, :]))
    
    is_dominated = (at_least_as_good & strictly_better).any(axis=0)

    # ties survive together, since neither dominates the other
    return sweep_df[~is_dominated].copy()

def constrained_thresholds(frontier_df, risk_grid):
    """Fastest frontier pair under each shared risk cap; NaN where none qualify."""
    rows = []
    for max_risk in risk_grid:
        feasible = frontier_df[(frontier_df.ia_risk <= max_risk) & (frontier_df.copy_risk <= max_risk)]
        if len(feasible) == 0:
            rows.append((max_risk, np.nan, np.nan, np.nan))
            continue
        best = feasible.loc[feasible.speedup.idxmax()]
        rows.append((max_risk, best.t_l, best.t_h, best.speedup))
    return pd.DataFrame(rows, columns=["max_risk", "t_l", "t_h", "speedup"])

def rules_mask(frontier_df, rules):
    """Boolean mask: which frontier rows satisfy the rules (ia_risk/copy_risk/optional min_speedup)."""
    mask = (frontier_df.ia_risk <= rules["max_ia_risk"]) & (frontier_df.copy_risk <= rules["max_copy_risk"])
    if rules.get("min_speedup") is not None:
        mask &= frontier_df.speedup >= rules["min_speedup"]
    return mask

def frontier_with_validity(frontier_df, rules):
    """Full frontier tagged 'valid'/'invalid' against the rules; valid rows first,
    each half sorted by speedup desc."""
    annotated = frontier_df.assign(valid=rules_mask(frontier_df, rules))
    valid_rows = annotated[annotated.valid].sort_values("speedup", ascending=False)
    invalid_rows = annotated[~annotated.valid].sort_values("speedup", ascending=False)
    return pd.concat([valid_rows, invalid_rows], ignore_index=True)

def frontier_csv_name(rules):
    """Filename encoding the rules that determined validity for this file."""
    name = f"rule_ia{rules['max_ia_risk']}_copy{rules['max_copy_risk']}_frontier"
    if rules.get("min_speedup") is not None:
        name += f"_minspeedup{rules['min_speedup']}"
    return name + ".csv"

def select_thresholds(frontier_df, rules):
    """Fastest frontier pair satisfying the rules, or None if none qualify."""
    feasible = frontier_df[rules_mask(frontier_df, rules)]
    if len(feasible) == 0:
        return None
    return feasible.loc[feasible.speedup.idxmax()]

def region_deltas(tab, cfg, t_l, t_h):
    """Mean delta mIoU/accuracy in the IA region and the copy region, plus each
    region's sample size -- a mean over a handful of frames isn't a stable estimate,
    so the sample size needs to travel with it, not be left implicit."""
    score = tab.score.values
    f1_drop = (tab.previous_f1 - tab.current_f1).values
    miou_drop = (tab.previous_miou - tab.current_miou).values

    num_fresh = int((score < t_l).sum())

    copy_mask = score >= t_h
    num_copy = int(copy_mask.sum())
    delta_miou_copy = miou_drop[copy_mask].mean() if num_copy else np.nan
    delta_accuracy_copy = f1_drop[copy_mask].mean() if num_copy else np.nan

    ia_tab = tab.dropna(subset=["loss_ia"])
    ia_mask = (ia_tab.score.values >= t_l) & (ia_tab.score.values < t_h)
    num_ia = int(ia_mask.sum())
    delta_miou_ia = ia_tab.miou_gap.values[ia_mask].mean() if num_ia else np.nan
    delta_accuracy_ia = ia_tab.f1_gap.values[ia_mask].mean() if num_ia else np.nan

    return delta_miou_ia, delta_accuracy_ia, delta_miou_copy, delta_accuracy_copy, num_ia, num_copy, num_fresh

def frontier_with_deltas(frontier_df, tab, cfg):
    """Frontier rows annotated with mean delta mIoU/accuracy and sample size in each region."""
    deltas = [region_deltas(tab, cfg, row.t_l, row.t_h) for row in frontier_df.itertuples()]
    delta_miou_ia, delta_accuracy_ia, delta_miou_copy, delta_accuracy_copy, n_ia, n_copy, n_fresh = zip(*deltas)
    return frontier_df.assign(delta_miou_ia=delta_miou_ia, delta_accuracy_ia=delta_accuracy_ia,
                               delta_miou_copy=delta_miou_copy, delta_accuracy_copy=delta_accuracy_copy,
                               n_fresh=n_fresh,
                               n_ia=n_ia, n_copy=n_copy)

def sweep_rules(tab, cfg, frontier_df, rules_sweep):
    rows = []
    deltas_cache = {}

    def cached_deltas(t_l, t_h):
        if (t_l, t_h) not in deltas_cache:
            deltas_cache[(t_l, t_h)] = region_deltas(tab, cfg, t_l, t_h)
        return deltas_cache[(t_l, t_h)]

    for max_ia_risk, max_copy_risk in itertools.product(rules_sweep["max_ia_risk_grid"],
                                                          rules_sweep["max_copy_risk_grid"]):
        risk_feasible = frontier_df[(frontier_df.ia_risk <= max_ia_risk) &
                                     (frontier_df.copy_risk <= max_copy_risk)]
        if len(risk_feasible) == 0:
            continue

        for min_speedup in rules_sweep["min_speedup_grid"]:
            rules = dict(max_ia_risk=max_ia_risk, max_copy_risk=max_copy_risk, min_speedup=min_speedup)
            selected = select_thresholds(risk_feasible, rules)
            if selected is None:
                continue

            delta_miou_ia, delta_accuracy_ia, delta_miou_copy, delta_accuracy_copy, n_ia, n_copy, n_fresh = \
                cached_deltas(selected.t_l, selected.t_h)
            rows.append((max_ia_risk, max_copy_risk, min_speedup,
                         selected.t_l, selected.t_h, selected.speedup, selected.ia_risk, selected.copy_risk,
                         delta_miou_ia, delta_accuracy_ia, delta_miou_copy, delta_accuracy_copy,
                         n_fresh, n_ia, n_copy))

    columns = ["max_ia_risk", "max_copy_risk", "min_speedup",
               "t_l", "t_h", "speedup", "ia_risk", "copy_risk",
               "delta_miou_ia", "delta_accuracy_ia", "delta_miou_copy", "delta_accuracy_copy",
               "n_fresh", "n_ia", "n_copy"]
    return pd.DataFrame(rows, columns=columns)

# PLOT
def plot_threshold_curves(t_h_curve, t_l_curve, ref_t_l, ref_t_h, rules, cfg, out_path):
    """Four panels: t_h swept (t_l fixed) and t_l swept (t_h fixed), score on x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    best_t_h_row = t_h_curve[t_h_curve.threshold == ref_t_h].iloc[0]
    best_t_l_row = t_l_curve[t_l_curve.threshold == ref_t_l].iloc[0]

    def plot_pair(axis, column):
        axis.plot(t_h_curve.threshold, t_h_curve[column], color="tab:orange", label=f"t_h (t_l fixed = {ref_t_l:.2f})")
        axis.plot(t_l_curve.threshold, t_l_curve[column], color="tab:blue", label=f"t_l (t_h fixed = {ref_t_h:.2f})")
        axis.scatter([ref_t_h], [best_t_h_row[column]], s=40, color="tab:orange",
                     edgecolor="black", linewidth=0.6, zorder=5, label=f"best t_h = {ref_t_h:.2f}")
        axis.scatter([ref_t_l], [best_t_l_row[column]], s=40, color="tab:blue",
                     edgecolor="black", linewidth=0.6, zorder=5, label=f"best t_l = {ref_t_l:.2f}")
        axis.set_xlabel("score threshold")

    plot_pair(axes[0, 0], "risk_cap")
    axes[0, 0].axhline(rules["max_copy_risk"], color="k", lw=.7, ls=":", label="max_copy_risk rule")
    axes[0, 0].axhline(rules["max_ia_risk"], color="gray", lw=.7, ls=":", label="max_ia_risk rule")
    axes[0, 0].set_ylabel("risk tolerance cap")
    axes[0, 0].legend(fontsize=7)

    plot_pair(axes[0, 1], "speedup")
    axes[0, 1].set_ylabel("speedup (time-weighted)")
    axes[0, 1].legend(fontsize=7)

    plot_pair(axes[1, 0], "delta_miou")
    axes[1, 0].axhline(cfg["TAU"], color="k", lw=.7, ls=":", label="TAU")
    axes[1, 0].set_ylabel("delta mIoU (mean)")
    axes[1, 0].legend(fontsize=7)

    plot_pair(axes[1, 1], "delta_accuracy")
    axes[1, 1].axhline(cfg["TAU"], color="k", lw=.7, ls=":", label="TAU")
    axes[1, 1].set_ylabel("delta accuracy / F1 (mean)")
    axes[1, 1].legend(fontsize=7)

    fig.suptitle("Threshold trade-offs vs score threshold: t_h sweeps the copy cutoff, t_l sweeps the IA cutoff")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

def plot_thresholds_vs_tolerance(constrained_df, out_path):
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(constrained_df.max_risk, constrained_df.t_l, marker="o", markersize=2, label="t_l (lower)")
    axis.plot(constrained_df.max_risk, constrained_df.t_h, marker="o", markersize=2, label="t_h (upper)")
    axis.fill_between(constrained_df.max_risk, constrained_df.t_l, constrained_df.t_h,
                       color="gray", alpha=0.12, label="IA reuse band")
    axis.set_xlabel("risk tolerance cap (max allowed ia_risk and copy_risk)")
    axis.set_ylabel("score threshold")
    axis.set_title("Best-achievable thresholds as risk tolerance loosens")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

# MAIN
def run_combo(base_cfg, combo):
    """Run the full threshold-selection pipeline for one (model, dataset) combo."""
    print(f"\n{'=' * 60}\n{combo['model']} + {combo['oracle_dataset']}\n{'=' * 60}")

    cfg = dict(base_cfg, **combo)
    cfg["out_dir"] = os.path.join(base_cfg["out_dir"], f"{combo['model']}_{combo['oracle_dataset']}")
    sheet_name_prefix = combo['model'] + '+' + combo['oracle_dataset'] + '_'

    tab = load(cfg)
    cfg["GRID"] = SCORE_GRID(tab, cfg["MIN_SCORE_THRESHOLD"])

    if cfg.get("save_joined_table"):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        joined_table_csv = os.path.join(cfg["out_dir"], cfg["joined_table_csv"])
        tab.to_csv(joined_table_csv, index=False)
        sync_csv_to_sheet(joined_table_csv, sheet_name_prefix=sheet_name_prefix)
        print(f"joined table: {joined_table_csv}")

    rules = rules_with_overrides(RULES)

    sweep_df = sweep(tab, cfg)
    frontier_df = pareto_front(sweep_df)
    selected = select_thresholds(frontier_df, rules)

    reference = selected if selected is not None else frontier_df.loc[frontier_df.speedup.idxmax()]
    ref_t_l, ref_t_h = reference.t_l, reference.t_h
    t_h_curve = threshold_curve_by_t_h(tab, cfg, ref_t_l)
    t_l_curve = threshold_curve_by_t_l(tab, cfg, ref_t_h)

    print(f"joined rows: {len(tab)}")
    print(f"swept {len(sweep_df)} (t_l, t_h) pairs; {len(frontier_df)} on the Pareto frontier")

    if selected is None:
        print("selected: none -- no frontier pair satisfies these rules")
    else:
        print(f"selected: t_l={selected.t_l}  t_h={selected.t_h}  speedup={selected.speedup:.3f}  "
              f"ia_risk={selected.ia_risk:.3f}  copy_risk={selected.copy_risk:.3f}")

    # get frontier
    frontier_display = frontier_with_validity(frontier_with_deltas(frontier_df, tab, cfg), rules)

    # best t_l/t_h across the full range of a shared risk cap
    constrained_df = constrained_thresholds(frontier_df, cfg["RISK_TOLERANCE_GRID"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    curves_png = os.path.join(cfg["out_dir"], cfg["curves_plot_png"])
    tolerance_png = os.path.join(cfg["out_dir"], cfg["tolerance_plot_png"])
    frontier_csv = os.path.join(cfg["out_dir"], frontier_csv_name(rules))
    frontier_display.to_csv(frontier_csv, index=False)
    sync_csv_to_sheet(frontier_csv, sheet_name_prefix=sheet_name_prefix)

    plot_threshold_curves(t_h_curve, t_l_curve, ref_t_l, ref_t_h, rules, cfg, curves_png)
    plot_thresholds_vs_tolerance(constrained_df, tolerance_png)

    rules_sweep_df = sweep_rules(tab, cfg, frontier_df, RULES_SWEEP)
    rules_sweep_csv = os.path.join(cfg["out_dir"], cfg["rules_sweep_csv"])
    rules_sweep_df.to_csv(rules_sweep_csv, index=False)

    # sync to google sheet
    sync_csv_to_sheet(rules_sweep_csv, sheet_name_prefix=sheet_name_prefix)

    return dict(model=combo["model"], dataset=combo["oracle_dataset"], joined_rows=len(tab), selected=selected)

if __name__ == "__main__":
    summaries = [run_combo(CONFIG, combo) for combo in COMBOS]

    summary_rows = [(s["model"], s["dataset"], s["joined_rows"],
                      s["selected"].t_l if s["selected"] is not None else None,
                      s["selected"].t_h if s["selected"] is not None else None,
                      s["selected"].speedup if s["selected"] is not None else None)
                     for s in summaries]
    summary_df = pd.DataFrame(summary_rows, columns=["model", "dataset", "joined_rows", "t_l", "t_h", "speedup"])
    summary_csv = os.path.join(CONFIG["out_dir"], "summary_all_combos.csv")
    summary_df.to_csv(summary_csv, index=False)

    # sync to google sheet    
    sync_csv_to_sheet(summary_csv)

    print(f"\n{'=' * 60}\nall combos\n{'=' * 60}")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nsummary: {summary_csv}")
