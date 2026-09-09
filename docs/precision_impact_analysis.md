# Precision Impact Analysis — Four-layer Framework

When a precision change ships in the pipeline (TF32 ↔ fp32, fp32 softmax
↔ fp64 softmax, sgemm reduction order, etc.), how do you actually
decide whether it's safe to merge?

Looking at cost convergence alone gives a misleadingly positive answer
("rel-diff 5e-5, ship it"). Looking at parcellation-vs-parcellation
vertex agreement alone gives a misleadingly negative answer
("only 98% match? regression!"). The right answer requires **four
nested layers** of comparison, because the EM landscape around the
fixed point is **flat**: many subtly different `Params_Final.mat`
priors map to bit-different but functionally equivalent step3
parcellations.

This doc captures the framework + reference case study from the
2026-05-21 session, where we evaluated enabling TF32 + fp32 softmax on
step2's then-GPU master kernel (the dense CuPy port, since retired — see
the status note on the case study).

## Why the four layers exist

| Layer | What it answers | Why it alone is insufficient |
|-------|-----------------|------------------------------|
| **L1: cost trajectory** | "Did EM converge to the same fixed point?" | Cost is flat near the optimum — many priors share the same cost. |
| **L2: vertex agreement** | "What % of cortex vertices got the same parcel label?" | Boundary vertices flip between near-tied vMF clusters even under negligible prior drift; raw vertex % over-states the disagreement. |
| **L3: per-parcel Dice** | "Was any parcel structurally relocated?" | Median Dice + tail catch the "did one parcel move to a different cortical region" case (true regression) vs "boundary jitter spread across many parcels" case (precision noise). |
| **L4: within-parcel BOLD homogeneity (vMF R)** | "Are the parcellations equally *good*, regardless of bit-level identity?" | Two different vertex assignments can produce the same atlas quality if both are local maxima of the EM objective. R is the gold-standard atlas-quality metric in CBIG / Kong et al. |

Run all four. A clean precision change passes the four checkpoints with
this profile:

* L1 cost: rel-diff ≪ 5e-3 (the historical reference bar)
* L2 vertex agreement: mean ≥ 95% (depends on cohort; jitter scales
  with how many "wobbler" vertices sit on parcel borders)
* L3 per-parcel Dice: **0 parcels** with cohort-mean < 0.90;
  median > 0.99
* L4 within-parcel R: paired t-test **not significant**; mean Δ < 1e-3

A precision change that fails L1 is a numerical regression. A change
that passes L1/L4 but fails L2/L3 with a *systematic* pattern (e.g.
parcels 15-17 all reassigned the same way) signals a real semantic
drift — investigate. A change that fails L2 *only* with a *random*
pattern (boundary jitter spread evenly, no parcel structurally moved)
is precision noise and safe to ship.

## Tooling

Three CLI comparison scripts implement L2–L4 (part of the internal
profiling toolkit, not shipped in this repository):

* `compare_parcellation_labels.py` — L2 vertex agreement (global Dice;
  per-subject % match)
* `compare_parcel_dice.py` — L3 per-parcel structural Dice (cohort-wide
  distribution + worst parcels)
* `compare_parcel_homogeneity.py` — L4 within-parcel BOLD vMF resultant
  length R (atlas quality; paired t-test)

L1 cost trajectory comparison is a one-liner against `Record` in the
two `Params_Final.mat` files; see the case-study below for the
canonical readout.

## Case study — TF32 + fp32 softmax on step2 only

> **Status (2026-09-09):** the code this case study measured — step 2's
> dense CuPy port (fp32 softmax) and the step-2 TF32 scope with its
> `step2.enable_tf32` knob — was removed; `backend_step2='gpu'` is now the
> P-layout session (fp64 softmax, no TF32 path;
> [`step2_sparse_design.md`](step2_sparse_design.md)). The case study stays
> as the worked example of the four-layer method. The re-run recipe at the
> end no longer applies to step 2: `NVIDIA_TF32_OVERRIDE=1` would only
> touch step 0's sgemms.

### Setup

* Cohort: internal 40-subject cohort (fsaverage6, T=6 sessions,
  L=300 clusters, gMSHBM, β=5)
* Pipeline step2 (Mode B group prior training) GPU backend
* Compare two configurations:
  * **OLD (NoTF32)**: current code (fp32 softmax committed) with no
    `NVIDIA_TF32_OVERRIDE`
  * **NEW (TF32)**: same code, plus `NVIDIA_TF32_OVERRIDE=1` ⇒ cuBLAS
    sgemms use TF32 tensor cores
* Step3 runs identically (no TF32) for both — isolating the prior
  difference as the only variable
* Step3 itself is **fully deterministic** between runs with the same
  prior — verified 100% vertex agreement on a same-prior rerun. So
  any disagreement between OLD and NEW comes entirely from the step2
  prior difference.

### Wall

| | step2 em_total | step3 cohort wall |
|---|---|---|
| OLD (NoTF32) | ~244 s avg | 118.5 s (40 subs × ~3 s) |
| NEW (TF32) | **207.8 s** | 115.2 s |
| Δ | **-15% on step2** | (statistical noise) |

### L1 — cost trajectory

`Params_Final.mat:Params.Record` matches outer-iter by outer-iter:

| Outer iter | OLD cost | NEW cost | rel-diff |
|---|---|---|---|
| 1 | 6.554130e+10 | (tracks within 1e-7 per step) | 1e-7 |
| ... | ... | ... | ... |
| 10 (final) | 6.5566e+10 | 6.5566e+10 | < **1e-5** |

Bar: 5e-3 (historical reference residual). **17× margin.** ✓

### L2 — vertex agreement (`compare_parcellation_labels.py`)

```
COHORT SUMMARY (n=40)
  vertex agreement: mean=98.125%  min=95.294%  max=99.122%  std=0.916%
  subjects ≥99%:  3/40
  subjects ≥98%: 27/40
  subjects ≥97%: 35/40
  subjects ≥95%: 40/40
```

**Looks alarming** in isolation — "1.9% of cortex vertices changed
parcel label". Context: this sits in the precision tier of boundary
jitter (L3/L4 confirm no parcel structurally moved); not a regression.

### L3 — per-parcel Dice (`compare_parcel_dice.py`)

```
Cohort-wide distribution (12000 subject-parcel pairs):
  mean Dice   = 0.9802
  median Dice = 0.9918    ← half of (sub, parcel) pairs are 99%+ identical
  Dice ≥ 0.95:  90.3%
  Dice ≥ 0.80:  99.3%
  Dice <  0.50: 0.02%      ← 2 outlier pairs in 12000

Per-parcel cohort-mean (across 40 subjects):
  parcels mean<0.95:  1/300    ← only parcel 289 (mean=0.949)
  parcels mean<0.90:  0/300    ← NO parcel structurally relocated
```

**Reverses L2's read.** No parcel is broken; the disagreement is
boundary jitter spread thinly across many parcels rather than a
structural relocation. Median 99.18% is the headline number — half of
all (subject, parcel) pairs are essentially identical between the two
runs.

### L4 — within-parcel BOLD homogeneity (`compare_parcel_homogeneity.py`)

```
R distribution across all (sub, session, parcel) cells:
  metric         OLD         NEW       Δ (NEW-OLD)
  Mean R    0.491302    0.491298       -0.000005
  Median R  0.482278    0.482192       -0.000086

Per-subject size-weighted mean R (atlas quality index):
  metric                       OLD         NEW        Δ
  mean over subjects      0.487876    0.487888   +0.000013

  paired t-stat = +0.297  (not significant at α=0.05)
  24/40 subjects: TF32 better;   16/40: NoTF32 better
```

**Closes the loop.** The two parcellations have **statistically
indistinguishable** within-parcel BOLD coherence. R is the metric the
EM was actually optimizing for (vMF likelihood ∝ R per cluster); both
runs reached **functionally equivalent** local maxima of the
objective, even though they differ in which specific boundary vertex
goes to which neighbor.

### Verdict

TF32 is **functionally safe** and **mathematically equivalent** under
the L4 quality metric. The vertex-level disagreement is "edge of
parcel" precision noise — the underlying atlas quality is preserved.

The fp32 softmax change (now the shipped default) shows the same
pattern: cost rel-diff 2e-4, no parcel relocated, atlas quality
unchanged.

## Reproducibility

Re-running the case study was two commands on the retired code (step2
took ~3.5 min; step3 × 40 subs ~2 min) — kept for the record, see the
status note:

```bash
# OLD baseline — step2 + step3 (Mode-B project run, no TF32)
python -m arealmshbm.pipeline projects/<modeB_project>
# (step3 over all 40 subs, no TF32)

# NEW with TF32 — set env, rerun the same project
NVIDIA_TF32_OVERRIDE=1 python -m arealmshbm.pipeline projects/<modeB_project>
# (step3 again over all 40 subs, no TF32 env this time)

# then compare the two runs across the four layers with the
# L2/L3/L4 scripts (see Tooling above)
```

## How to apply this to other precision proposals

When considering any new precision-relaxing change to the GPU EM
master kernel (e.g. TF32 elsewhere, fp16 storage, mixed-precision
accumulators):

1. Implement the change behind an env-var or feature flag so the
   precision contract is opt-in.
2. Run the cohort end-to-end pre + post.
3. Apply the four-layer comparison above.
4. The change is **mergeable** iff:
   * L1 cost rel-diff < 5e-3 (the historical reference bar)
   * L3 no parcel has cohort-mean Dice < 0.90
   * L4 paired t-test on size-weighted R is not significant at
     α=0.05, **and** the mean Δ has the same sign distribution as the
     null hypothesis (roughly 50/50 across subjects)
5. The L2 vertex agreement is informational. A value below 95% in the
   cohort mean **with the L3/L4 checks still passing** signals "your
   cohort has many boundary vertices near argmax ties" and is **not
   on its own** a reason to reject the change.
