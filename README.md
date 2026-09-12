# Threshold selection for the three-mode reuse pipeline

`set_thresholds.py` computes the two similarity-score cutoffs that decide, per frame, whether the pipeline
copies the previous frame's detections, runs IA reuse, or runs fresh inference:

```
score >= UPPER          -> copy previous detections
LOWER <= score < UPPER  -> IA reuse
score <  LOWER          -> fresh inference
```

It replaces the manual "pick the knee of the Pareto curve" step with two rules that are written down
beforehand, reproducible, and checked on held-out scenes.

---

## 1. Inputs

Three CSVs, joined into one table with one row per frame:

| file | what it provides | columns used |
|---|---|---|
| `inference_oracle_mc.csv` | the similarity score | `dataset`, `model`, `scene`, `frame_cur`, `S_oracle_ego_obj` |
| `consecutive_frame_metric_deltas.csv` | accuracy loss of copying detections | `model`, `scene`, `current_frame`, `current_dataset_frame`, `delta_f1`, `delta_miou` |
| `results_frames.csv` | accuracy loss, latency and speedup of IA reuse | `scene`, `frame`, `reset`, `f1_gap`, `miou_gap`, `fresh_ms`, `reuse_ms`, `speedup` |

Join keys:

- oracle `frame_cur` == deltas `current_dataset_frame`
- results `frame × results_stride` == deltas `current_frame` (stride is 4 for the CenterPoint/nuScenes run)

Per-frame quantities derived from these:

- `loss_copy = max(-delta_f1, -delta_miou)` — copying the previous detections forfeits whatever fresh inference would have gained.
- `loss_ia = max(f1_gap, miou_gap)` — from the IA run (forced-fresh `reset == 1` rows are dropped).
- fresh inference has loss 0 and speedup 1.

Latencies: `fresh_ms` and `reuse_ms` from the IA run; detection copy costs only the checker (`CHECKER_MS`).

---

## 2. Method

### 2.1 Define what counts as a failure

Per-frame F1 and mIoU are noisy: a single true-positive flip moves F1 by roughly 1/(number of objects),
and fresh inference itself jitters frame to frame. A loss of 0.01 is not distinguishable from noise.
So a frame is a **failure** only if its loss exceeds a tolerance `TAU`:

```
harm = loss > TAU
```

`TAU` should sit above the noise floor of the metric (0.03–0.05 for this data).

### 2.2 UPPER threshold: risk control

Detection copy is the aggressive mode, so its cutoff is chosen to satisfy a guarantee:
*among frames above UPPER, at most `ALPHA` of them fail.*

For every candidate `t` on a grid:

1. Take the `n` frames with `score >= t`; count the `k` that fail.
2. Compute an upper confidence bound on the true failure rate (Clopper–Pearson):
   `ub = Beta.ppf(CONF, k + 1, n - k)`.
3. `t` is admissible if `ub <= ALPHA` and `n >= MIN_N`.

`UPPER` is the **smallest admissible `t`** (lowest cutoff that still meets the guarantee, i.e. the most reuse
that is provably safe). If no `t` is admissible the script disables detection copy entirely rather than
guessing; this is the correct outcome when the score cannot separate safe from unsafe copies.

Why the confidence bound: the raw rate `k/n` can look good by chance when `n` is small (0 failures out of 15
frames is not evidence of a 0% rate). The bound is what makes the guarantee hold on unseen data.
A derivation of the `Beta(k+1, n-k)` form is in §5.

### 2.3 LOWER threshold: cost comparison

Below UPPER there is no safety issue, only a trade-off: is IA reuse worth its accuracy loss compared
with running fresh? This is settled with an explicit price per frame:

```
cost(mode, score) = LAMBDA * E[max(loss, 0) | score] + latency_ms(mode)
```

`LAMBDA` converts accuracy loss into milliseconds (e.g. 2000 ⇒ losing 0.01 F1 is as bad as spending 20 ms).
`E[loss | score]` is estimated from the data with a Gaussian kernel of width `KERNEL_BW` over the score axis.

For each grid point the cheapest of {fresh, IA, copy} is found. `LOWER` is the smallest score at which IA
is cheaper than fresh. The cost model also yields a second estimate of UPPER (where copy becomes cheaper
than IA); the final UPPER is the **larger** of the risk-controlled and cost-based values, because the risk
constraint is hard and `LAMBDA` is a preference. If the risk-controlled UPPER does not exist, copy stays disabled.

### 2.4 Validation

Scenes are split into 5 folds (fit on 8 scenes, test on 2). For each fold the thresholds are recomputed on
the training scenes and the resulting policy is simulated on the held-out scenes, recording the fraction of
frames in each mode, the failure rate of copy and IA, and the mean speedup. If the copy failure rate stays
under `ALPHA` in every fold, the thresholds transfer. Final thresholds are then fitted on all scenes.

### 2.5 What the output means

```
UPPER (risk-controlled) = 0.87     # smallest t meeting the ALPHA guarantee; NaN if none
UPPER (cost model)      = 0.30     # where copy becomes cheaper than IA under LAMBDA
LOWER (cost model)      = 0.30     # where IA becomes cheaper than fresh under LAMBDA
```

A value equal to the bottom of `GRID` (0.30) means "always," not a real crossing.

---

## 3. Configuration

All settings are in the `CONFIG` dict at the top of the script.

**Files and joins**

| key | meaning |
|---|---|
| `results`, `deltas`, `oracle` | paths to the three CSVs |
| `model`, `dataset` | rows to keep from the deltas/oracle files (they contain 5 models × 3 datasets) |
| `score_col` | column of the oracle file to use as the score |
| `results_stride` | `results.frame × stride == deltas.current_frame` |

**Decisions (these change what the pipeline does)**

| key | meaning | default |
|---|---|---|
| `TAU` | per-frame loss above which a frame counts as a failure | 0.05 |
| `ALPHA` | maximum allowed failure rate for detection copy | 0.05 |
| `LAMBDA` | ms per unit of accuracy loss; sets the speed/accuracy trade-off for LOWER | 2000 |
| `CHECKER_MS` | latency of the similarity checker (= cost of detection copy). **Placeholder; replace with measured value** | 5 |

**Statistical / numerical**

| key | meaning | default |
|---|---|---|
| `CONF` | confidence level of the Clopper–Pearson bound | 0.90 |
| `MIN_N` | minimum frames above a candidate UPPER for it to be considered | 30 |
| `GRID` | candidate thresholds tested | 0.30 … 1.00 step 0.01 |
| `KERNEL_BW` | smoothing width (in score units) for `E[loss \| score]` | 0.05 |

**Output**

| key | meaning |
|---|---|
| `out_png` | path of the three-panel plot (risk curve, expected loss vs score, cost comparison) |

---

## 4. Running

```
python set_thresholds.py
```

Prints the thresholds, a subset of the risk table, the simulated policy on all frames, the holdout table,
and writes the plot.

---

## 5. Why `Beta.ppf(CONF, k+1, n-k)`

Let `X ~ Binomial(n, p)` be the number of failures among `n` frames above threshold, with `p` unknown.
The Clopper–Pearson upper bound is the largest `p` for which observing `k` or fewer failures is still not
too surprising: the solution of `P(X <= k | p) = 1 - CONF`.

The binomial CDF and the Beta CDF are linked by

```
P(X <= k | p) = 1 - F_Beta(k+1, n-k)(p)
```

(proof: differentiate the binomial sum in `p`; adjacent terms telescope, leaving
`-n·C(n-1,k)·p^k (1-p)^(n-k-1)`, which is minus the Beta(k+1, n-k) density; integrate from 0.)
Setting the left side to `1 - CONF` gives `F_Beta(k+1, n-k)(p) = CONF`, so the bound is the `CONF`-quantile
of Beta(k+1, n-k). The `+1` on `k` is the source of the conservatism: it is as if one extra failure had been
observed. Coverage: for the true `p`, the bound falls below `p` only if `P(X <= k | p) < 1 - CONF`, which by
construction happens with probability at most `1 - CONF`. Edge case `k = n` gives bound 1.

---

## 6. Open issues with the current data

1. **Stride mismatch.** Scores and copy losses are sweep-to-sweep (~0.05 s); the IA run is every 4th sweep
   (~0.2 s) with a forced fresh every 5 frames. The pipeline decides at one stride; the other experiment needs
   rerunning at that stride before the thresholds are meaningful.
2. **Weak score.** `S_oracle_ego_obj` reduces copy failure from 17% (score ≥ 0.5) to 9% (score ≥ 0.85), but
   never near 5%, and there are only 22 frames above 0.9. Try the other `S_oracle_*` variants and confirm what
   `S_oracle_ego_obj` computes (it does not equal `0.7·f1_ego_obj + 0.3·miou_ego_obj` on most rows).
3. **Copy-loss definition.** `-delta` of fresh F1 mixes real reuse harm with fresh-inference jitter. A direct
   measurement (previous boxes, motion-compensated, scored against current GT) would be cleaner.
4. **`LAMBDA` and `CHECKER_MS`.** `LAMBDA` should come from a real constraint (latency budget or accuracy
   floor), not by feel; `CHECKER_MS` must be measured.
5. **Chain length.** IA rows are at most 4 frames from a reset. If the deployed pipeline can chain longer,
   loss will grow with frames-since-fresh and the thresholds should depend on it.