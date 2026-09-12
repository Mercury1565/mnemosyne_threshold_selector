import os
import numpy as np, pandas as pd
from scipy.stats import beta
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# CONFIG
CONFIG = dict(
    results = "./data/results_frames.csv",                    # IA reuse run
    deltas  = "./data/consecutive_frame_metric_deltas.csv",   # detection-copy deltas
    oracle  = "./data/inference_oracle_mc.csv",               # similarity score
    model   = "centerpoint",
    dataset = "nuscenes",
    score_col = "S_oracle_ego_obj",
    results_stride = 4,        # results_frames.frame * stride == deltas.current_frame

    TAU    = 0.05,             # per-frame loss tolerance (F1 / mIoU units)
    ALPHA  = 0.05,             # allowed failure rate for detection copy
    CONF   = 0.90,             # confidence level of the Clopper-Pearson bound
    LAMBDA = 2000.0,           # ms per unit of loss  (0.01 loss == 20 ms)
    CHECKER_MS = 5.0,          # TODO: replace with measured checker latency
    MIN_N  = 30,               # min frames above a threshold for it to count
    GRID   = np.round(np.arange(0.30, 1.001, 0.01), 3),
    KERNEL_BW = 0.05,          # smoothing width for E[loss | score]

    out_dir = "./outputs",
    risk_plot_png = "risk_control.png",
    loss_plot_png = "expected_loss.png",
    cost_plot_png = "cost_comparison.png",
)

# LOAD & JOIN
def load(cfg):
    results_df = pd.read_csv(cfg["results"])
    deltas_df = pd.read_csv(cfg["deltas"])
    oracle_df = pd.read_csv(cfg["oracle"])

    # filter to the model/dataset of interest
    deltas_df = deltas_df[deltas_df.model == cfg["model"]].copy()
    oracle_df = oracle_df[(oracle_df.model == cfg["model"]) & (oracle_df.dataset == cfg["dataset"])].copy()

    # score  <->  deltas   (oracle.frame_cur == deltas.current_dataset_frame)
    joined = deltas_df.merge(oracle_df[["scene", "frame_cur", cfg["score_col"]]],
                left_on=["scene", "current_dataset_frame"], right_on=["scene", "frame_cur"], how="inner")

    # rename score column
    joined = joined.rename(columns={cfg["score_col"]: "score"})

    # loss_copy proxy: there's no "frame t-1's boxes graded against frame t's ground truth" column
    # anywhere in this data, so copy risk is approximated by the frame-to-frame drop in each
    # frame's own independently-run fresh-inference score. A big drop is read as "the scene changed
    # enough that fresh inference itself got harder here", which we take as a signal that the
    # copied (stale) detections would have fared at least as badly.
    f1_drop = joined.previous_f1 - joined.current_f1
    miou_drop = joined.previous_miou - joined.current_miou
    joined["loss_copy"] = np.maximum(f1_drop, miou_drop)

    # IA reuse rows (subsampled), joined on results.frame * stride == deltas.current_frame
    results_df = results_df[results_df.reset == 0].copy()
    results_df["current_frame"] = results_df.frame * cfg["results_stride"]
    results_df["loss_ia"] = np.maximum(results_df.f1_gap, results_df.miou_gap)

    joined = joined.merge(results_df[["scene", "current_frame", "loss_ia", "fresh_ms", "reuse_ms", "speedup"]],
                on=["scene", "current_frame"], how="left")
    
    return joined

# HELPERS
def cp_upper(num_successes, num_trials, conf):
    """Clopper-Pearson upper bound on a binomial proportion."""
    if num_trials == 0 or num_successes >= num_trials:
        return 1.0
    return beta.ppf(conf, num_successes + 1, num_trials - num_successes)

def risk_table(score, harm, cfg):
    rows = []
    for threshold in cfg["GRID"]:
        mask = score >= threshold
        num_frames, num_harmful = int(mask.sum()), int(harm[mask].sum())
        rows.append((threshold, num_frames, num_harmful, num_harmful / num_frames if num_frames else np.nan,
                     cp_upper(num_harmful, num_frames, cfg["CONF"])))
    return pd.DataFrame(rows, columns=["thr", "n", "k", "risk", "risk_ucb"])

def upper_risk_controlled(score, harm, cfg):
    risk_df = risk_table(score, harm, cfg)
    valid_rows = risk_df[(risk_df.risk_ucb <= cfg["ALPHA"]) & (risk_df.n >= cfg["MIN_N"])]
    return (float(valid_rows.thr.min()) if len(valid_rows) else np.nan), risk_df

def exp_loss(score, loss, grid, bw):
    """Kernel-smoothed E[max(loss,0) | score = t] on a grid."""
    loss_pos = np.clip(loss, 0, None); results = []
    for threshold in grid:
        weights = np.exp(-0.5 * ((score - threshold) / bw) ** 2)
        results.append((weights * loss_pos).sum() / weights.sum() if weights.sum() > 1e-9 else np.nan)
    return np.array(results)

def cost_band(tab, cfg):
    """Thresholds from the 3-way cost comparison. Returns (lower, upper, frame)."""
    grid = cfg["GRID"]; lambda_cost = cfg["LAMBDA"]
    ia_rows = tab.dropna(subset=["loss_ia"])
    fresh_ms, ia_ms = ia_rows.fresh_ms.median(), ia_rows.reuse_ms.median()
    e_copy = exp_loss(tab.score.values, tab.loss_copy.values, grid, cfg["KERNEL_BW"])
    e_ia   = exp_loss(ia_rows.score.values,  ia_rows.loss_ia.values,   grid, cfg["KERNEL_BW"])

    cost_df = pd.DataFrame(
        dict(
            thr=grid,
            cost_fresh=fresh_ms,
            cost_ia=lambda_cost * e_ia + ia_ms,
            cost_copy=lambda_cost * e_copy + cfg["CHECKER_MS"],
            e_ia=e_ia, e_copy=e_copy
            )
        )
    
    best_idx = cost_df[["cost_fresh", "cost_ia", "cost_copy"]].values.argmin(axis=1)   # 0 fresh, 1 ia, 2 copy
    lower = float(grid[best_idx >= 1].min()) if (best_idx >= 1).any() else np.nan
    upper = float(grid[best_idx == 2].min()) if (best_idx == 2).any() else np.nan

    return lower, upper, cost_df

def select(tab, cfg):
    harm = (tab.loss_copy > cfg["TAU"]).values
    upper_rc, risk_df = upper_risk_controlled(tab.score.values, harm, cfg)
    lower_cost, upper_cost, cost_df = cost_band(tab, cfg)
    upper = np.nanmax([upper_rc, upper_cost]) if not (np.isnan(upper_rc) and np.isnan(upper_cost)) else np.nan

    if np.isnan(upper_rc):
        upper = np.nan   # the safety constraint is unmet: do not enable detection copy at all

    return dict(lower=lower_cost, upper=upper, upper_rc=upper_rc, upper_cost=upper_cost, risk=risk_df, cost=cost_df)

def evaluate(tab, lower, upper, cfg):
    """Simulate the policy on a table: mode per frame, harm rate of copy, mean speedup."""
    score = tab.score.values
    mode = np.where(score >= upper, "copy", np.where(score >= lower, "ia", "fresh")) if not np.isnan(upper) \
           else np.where(score >= lower, "ia", "fresh")
    copy_rows = tab[mode == "copy"]; ia_rows = tab[(mode == "ia")].dropna(subset=["loss_ia"])
    fresh_ms = tab.fresh_ms.median()
    speedup_arr = np.where(mode == "copy", fresh_ms / cfg["CHECKER_MS"],
         np.where(mode == "ia", tab.speedup.fillna(tab.speedup.median()), 1.0))
    
    return dict(frac_copy=(mode == "copy").mean(), frac_ia=(mode == "ia").mean(), frac_fresh=(mode == "fresh").mean(),
                copy_harm=(copy_rows.loss_copy > cfg["TAU"]).mean() if len(copy_rows) else np.nan,
                ia_harm=(ia_rows.loss_ia > cfg["TAU"]).mean() if len(ia_rows) else np.nan,
                mean_speedup=speedup_arr.mean())

def holdout(tab, cfg):
    scenes = sorted(tab.scene.unique()); rows = []
    for pair_start in range(0, len(scenes), 2):
        test_scenes = scenes[pair_start:pair_start + 2]
        train_tab = tab[~tab.scene.isin(test_scenes)]; test_tab = tab[tab.scene.isin(test_scenes)]
        selection = select(train_tab, cfg)
        eval_result = evaluate(test_tab, selection["lower"], selection["upper"], cfg)
        rows.append(dict(held_out=",".join(scene.replace("scene-", "") for scene in test_scenes),
                         lower=selection["lower"], upper=selection["upper"], **eval_result))
    return pd.DataFrame(rows)

# PLOTS
def _draw_threshold_lines(axis, selection):
    for threshold_val, color in [(selection["lower"], "orange"), (selection["upper"], "red")]:
        if not np.isnan(threshold_val):
            axis.axvline(threshold_val, color=color, ls=":", label=f"{color} threshold = {threshold_val:.2f}")

def plot_risk_control(risk_tab, selection, cfg, out_path):
    """Risk of a harmful copy vs. the candidate threshold, with its confidence bound."""
    fig, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot(risk_tab.thr, risk_tab.risk, label="P(copy loss > TAU | score >= threshold)")
    axis.plot(risk_tab.thr, risk_tab.risk_ucb, "--", label=f"{int(cfg['CONF'] * 100)}% upper confidence bound")
    axis.axhline(cfg["ALPHA"], color="k", lw=.7, label=f"ALPHA = {cfg['ALPHA']}")
    axis.set_xlabel("score threshold")
    axis.set_ylabel("P(copy loss > TAU)")
    axis.set_title("Upper threshold: risk control")
    _draw_threshold_lines(axis, selection)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

def plot_expected_loss(cost_tab, selection, cfg, out_path):
    """Kernel-smoothed expected loss of detection-copy vs. IA reuse, as a function of score."""
    fig, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot(cost_tab.thr, cost_tab.e_copy, label="E[copy loss | score]")
    axis.plot(cost_tab.thr, cost_tab.e_ia, label="E[IA loss | score]")
    axis.axhline(cfg["TAU"], color="k", lw=.7, label=f"TAU = {cfg['TAU']}")
    axis.set_xlabel("score")
    axis.set_ylabel("expected loss (F1 / mIoU units)")
    axis.set_title("Expected loss vs score")
    _draw_threshold_lines(axis, selection)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

def plot_cost_comparison(cost_tab, selection, cfg, out_path):
    """Modeled cost (ms-equivalent) of fresh inference vs. IA reuse vs. detection copy."""
    fig, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot(cost_tab.thr, cost_tab.cost_fresh, label="fresh")
    axis.plot(cost_tab.thr, cost_tab.cost_ia, label="IA")
    axis.plot(cost_tab.thr, cost_tab.cost_copy, label="copy")
    axis.set_xlabel("score threshold")
    axis.set_ylabel("cost (ms-equivalent)")
    axis.set_title("Lower threshold: cost comparison")
    _draw_threshold_lines(axis, selection)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

# MAIN
if __name__ == "__main__":
    cfg = CONFIG
    tab = load(cfg)

    print(f"joined rows: {len(tab)}  (with IA data: {tab.loss_ia.notna().sum()})  scenes: {tab.scene.nunique()}")
    print(f"TAU={cfg['TAU']}  ALPHA={cfg['ALPHA']}  LAMBDA={cfg['LAMBDA']} ms/loss  checker={cfg['CHECKER_MS']} ms")
    print(f"base rates: P(copy loss>TAU)={(tab.loss_copy>cfg['TAU']).mean():.3f}   "
          f"P(IA loss>TAU)={(tab.loss_ia.dropna()>cfg["TAU"]).mean():.3f}")

    selection = select(tab, cfg)
    print("\n--- thresholds on all scenes ---")
    print(f"UPPER (risk-controlled) = {selection['upper_rc']}")
    print(f"UPPER (cost model)      = {selection['upper_cost']}")
    print(f"LOWER (cost model)      = {selection['lower']}")
    print(f"=> LOWER={selection['lower']}  UPPER={selection['upper']}")

    if np.isnan(selection["upper_rc"]):
        print("   ! no threshold satisfies the ALPHA constraint: detection copy disabled; policy is IA/fresh only")

    print("\nrisk table (subset):")
    print(selection["risk"][selection["risk"].thr.isin([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])].to_string(index=False))
    print("\npolicy on all frames:", {key: round(value, 3) for key, value in evaluate(tab, selection["lower"], selection["upper"], cfg).items()})

    print("\n--- scene-wise holdout (fit on 8, test on 2) ---")
    print(holdout(tab, cfg).round(3).to_string(index=False))

    # ---- plots (one file per plot)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    risk_png = os.path.join(cfg["out_dir"], cfg["risk_plot_png"])
    loss_png = os.path.join(cfg["out_dir"], cfg["loss_plot_png"])
    cost_png = os.path.join(cfg["out_dir"], cfg["cost_plot_png"])

    plot_risk_control(selection["risk"], selection, cfg, risk_png)
    plot_expected_loss(selection["cost"], selection, cfg, loss_png)
    plot_cost_comparison(selection["cost"], selection, cfg, cost_png)

    print(f"\nplots: {risk_png}, {loss_png}, {cost_png}")
