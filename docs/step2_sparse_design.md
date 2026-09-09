# Step-2 `gpu` backend — design contract (v2)

Working spec for the P-layout / bit-packed step-2 GPU backend (Mode-B
group-prior EM). It shipped in 2026-09 as the opt-in value
`backend='gpu_sparse'` beside the dense CuPy port that held `backend='gpu'`;
that port was removed on 2026-09-09 and this backend is now `backend='gpu'`
(the old name is rejected, not aliased). The **numerical reference is the
CPU backend** (`backend='cpu'`: the numba master in
`step2_em_iter_master/_kernels.py` plus the numba outer leaves in
`step2_em_outer/_kernels.py`), never that dense port. Module names follow
the per-leaf convention — `step2_em_iter_master/_kernels_gpu.py` (kernels)
and `session_gpu.py` (`Step2SparseSession`); the class and the `step2_io`
modules keep "sparse" in their names because it describes the layout.

v2 incorporates the design review: the CuPy flush-to-zero blocker (§1.10),
MATLAB-sparse masks (§5), active-support compaction promoted into v1 (§2.3),
K7 re-costed and re-implemented with cuBLAS (§3 K7), the fp32 kappa round-trip
(§1.4), and the smaller items in §8.

## 0. Why, and the measured starting point

Bench: `testdata/step2_bench/proj` — sub-001, fsaverage6, `N=81924 (n_lh=40962)`,
`T=6`, `D=1175` (profile length, `Db=147`), `dim=1174` (vMF dimension,
`= mtc.shape[0]-1`), `L=300 (L_lh=150)`, `D_grad=100`, gMSHBM `beta_internal=5000`,
`ini_val=403.611`. `testdata/step2_bench/proj2` is the same with S=2
(reference-cohort sub1+sub2). RTX 5090 Laptop (sm_120, 82 SMs, 64 MB L2, 24 GB), cupy 13.6, NVRTC 12.9.

| the dense CuPy port (retired 2026-09), fresh process, S=1 | wall |
|---|---:|
| load_inputs | 0.19-0.32 s (boundary .mat → dense (N,L) 0.11) |
| initialize_params.compose (CPU numba) | 0.52 s |
| session_ctor | 1.27 s fresh / 0.20 s warm |
| em_total (41 `run_iter` calls, mean m_iters 2.49, 133 ms each) | 3.7 s |
| CPU outer-EM closure (numba cache loads + leaves) | ~1.5 s fresh / 0.29 s warm |
| `Params_Final.mat` (dense fp64 compressed theta) | ~1.0 s |
| **total** | **6.9 s** (CPU backend 71 s; S=2: dense GPU 14.0 s, CPU 103 s) |

One EM iter (133 ms): fused E-step 87 ms (`cp.einsum` alone 72 ms), two
bitpacked→fp32 widens 23 ms, `X_dot_sl` sgemm 14.5, spatial_connect 4.4,
M-step 1.7, E.1 1.7, E.2 0.4.

Structure that makes the rewrite possible:

* `boundary_mask` (bm) is exactly 0/1, block-diagonal per hemisphere,
  `nnz = 586 522` (2.4 % of N·L, ≤ 15 per row, 1 695 empty rows, no empty
  columns; members per parcel 1154 / 1955 / 3381 min/mean/max). Define
  `P = {(n,l) : bm[n,l] != 0}`.
* `supp(s_lambda[s]) ⊆ P` always (E.1 multiplies by bm) and `supp(theta) ⊆ P`
  after the first E.2. `theta` is never reset and `theta = 0` is absorbing, so
  `supp(theta)` only shrinks; `supp(s_lambda[s]) ⊆ supp(theta)` from the second
  EM iter on. Converged `nnz(theta) = 73 655` (S=1) / `110 495` (S=2).
* BOLD is binary (density 9.7 %); `X[n,t,d] = (bit − mean)·inv` is never
  materialised — every contraction with `X` is a bit-sum plus a rank-1 term.
* **The old dense GPU port is wrong at S=1**: fp32 `exp` (CuPy is flush-to-zero,
  cliff at −87.3 vs −745 for the CPU's fp64) kills 5 528 alive vertices vs 1 303 on
  the CPU; cost −5 %, `kappa` 954 vs 922, 4 744 theta-argmax flips (S=2: 5 071).
  The rewrite restores the CPU semantics (fp64 `exp`, full-row `rmax` at iteration 1,
  subnormal-preserving fp32 stores).

Measured warm at S=1 (`step2_pipeline/profile.py`): EM iter ~3.1 ms,
`Step2Pipeline.run()` 0.32-0.36 s, run-to-run identical (`np.array_equal` on
every saved field), CPU-parity band far tighter than the old GPU's (§7, §9).

## 1. Numerics contract

Notation: `f32(x)` = round to fp32 (subnormal-aware, §1.10); ops written as the
CPU performs them. "exact" = bit-identical to the CPU reference given identical
inputs; "reassoc" = same value in exact arithmetic, different rounding order
(the CPU has no defined order there: MKL `sgemm`, or a tree we choose to keep).
Everything else is a listed deviation (§8). The RawModule is compiled with
`-std=c++17 -fmad=false`; no `--use_fast_math`; CUDA source ASCII only;
`INFINITY`/`NAN` macros are unavailable under NVRTC — use
`__int_as_float(0x7f800000)`, `__longlong_as_double(0x7ff0000000000000ULL)`,
`__int_as_float(0x7fc00000)`.

### 1.1 BOLD row statistics — exact

Per `(s,t,n)` row, `k = popcount(row)` (padding bits are zero by writer contract):

```
inv_D  = f32(1) / f32(D)                       (host constant, fp32 divide)
mean   = f32(k) * inv_D                        (fp32 multiply)  [NOT k/D]
dead   = (mean == 0.0f) || (mean == 1.0f)      (the CPU's per-cell has_zero gate, exactly; NOT k∈{0,D})
v_d    = f32(bit_d) - mean                     (fp32)
sumsq  = serial ascending-d fp32 sum of v_d*v_d  (fp32 multiply, fp32 add, no FMA)
inv    = dead ? 0 : f32(1) / f32(sqrt((double)sumsq))
```

Mirrors `step2_io/load_subject_profiles.py:27-118`. One thread per row.
`alive[s,t,n] = inv != 0`, `n_alive[s,n] = #{t : inv != 0}`. (For D=1175
`dead ⟺ k ∈ {0, D}`; the predicate form is used because `f32(D)·f32(1/D) == 1`
fails for 14 % of integers.) `inv = 0` makes `X ≡ 0` exactly as the CPU.

Consequences: alive rows of `X` have unit norm (up to one rounding),
`|lv_sum| ≤ n_alive`, and `tmp_idx[n] ⟺ n_alive[n] == 0` (R1 §4 Q1(b): exact
for `κ < Cdln(κ) ≈ 2308`; production κ ∈ [400, 1200]; warn once if `κ ≥ 2308`).

### 1.2 Init (`compose_init_state`) — exact

Per subject, per vertex `n` (CPU `step2_init/_kernels.py:40-102, 105-174`):

```
inv_T   = f32(1)/f32(T)
mean_d[d] = serial-over-t fp32 sum (start 0.0f) of X[n,t,d]   (X = f32(bit)-mean, then *inv: two fp32 ops)
a_d     = mean_d[d] * inv_T
norm_sq = serial ascending-d fp32 sum of a_d*a_d
acc[l]  = serial ascending-d fp32 sum of a_d * g[d,l]      (g = f32(mtc), (D,L); no FMA)
hard_label = first index of the strict maximum of acc (best=acc[0]; update on '>')
medial     = (norm_sq == 0)
s_lambda[s, n, l_act] = 1  iff  !medial and (n, l_act) ∈ P     (else the row stays 0)
theta[n,l] = f32( f64(inv_S_f32) + eps_f64 ) if any subject wrote (n,l) else f32(eps_f64)
theta_out  = f32(eps_f64) = 2.2204460e-16     (every cell outside P; §1.6)
```

`inv_S_f32 = f32(1)/f32(S)` (identical to the E.2 constant `f32(1.0/S)` for all
S ≤ 2000), `eps_f64 = 2.220446049250313e-16`. `initialize_state()` zero-fills
`s_lambda (S,P)`, `lv_sum`, `log_vmf`, `scr` before K1 and fills `log_theta`
from `theta` with the same kernel E.2 uses.

### 1.3 Phase A — reassoc

```
sigma_psi[s,l,d] = sigma[l] * s_psi[s,l,d]                                (exact)
X_dot_sl[s,t,l,d] = Σ_{n ∈ members(l)} s_lambda[s,n,l] · X[s,n,t,d]
                  = Σ_{n: bit(s,t,n,d)} w_n − Σ_n w_n · mean[s,t,n],   w_n = s_lambda[s,n,l] · inv[s,t,n]
```

`members(l)` = the **active** CSC column (§2.3). Port `x_dot_sl_bits`
(step-3 `m_step/m_step_gpu_sparse.py:133-240`) verbatim: block per `(s,t,l)`,
128 threads, members staged in shared memory in chunks of 64, fp32 `fmaf`
folds into an fp64 running sum, fp32 store. Measured 3.7-4.05 ms on the full P,
0.57 ms on the active support. `w_n` is a native fp32 multiply (FTZ, §1.10 —
accepted: a subnormal `s_lambda` contributes < 1e-38 relative).

### 1.4 Phase B — M-step

Per outer EM iter, once: `denom = T · Σ_{s,p} s_lambda[s,p]` in fp64 (fixed tree;
reassoc vs the CPU's serial loop, ~1e-16). Then per `iter_m` (1-based; loop until
`all_flag && drift < eps` or `iter_m > max_iter_m`, i.e. up to `max_iter_m + 1`
iterations):

```
kappa_sum = Σ_{s,t,l,d} s_t_nu_old · X_dot_sl        product fp32, accumulate fp64, fixed tree (reassoc)
rbar      = kappa_sum / denom
kappa     = invad(dim, rbar)                          device port of invad_numba (§3.1; see §8 for the secant-branch band)
if !isfinite(kappa): kappa = kappa_prev
if kappa < ini_val:  kappa = ini_val
kappa_f32 = f32(kappa)
col[d]    = kappa_f32 * X_dot_sl[s,t,l,d] + sigma_psi[s,l,d]     fp32 mul, fp32 add
cn        = sqrt( Σ_d col² )        fp64 accumulate (CPU: fp32 serial — deviation)
inv_cn    = f32(1.0 / cn)           (fp64 reciprocal; cn == 0 → +inf → stored NaN, IEEE)
old       = s_t_nu[..., d]          READ BEFORE THE WRITE (in-place buffer; §3 K4c)
new       = col[d] * inv_cn ;  s_t_nu[..., d] = new
cos       = Σ_d new * old           fp64 accumulate → f32 (CPU: fp32 serial — deviation)
converge[s,t,l] = (f32(1) - cos) < eps_f32           (NaN → false)
flag[s,t] = all_l converge          OVERWRITTEN every iter_m (NOT latched)
all_flag  = all_{s,t} flag
drift     = |kappa_prev - kappa| / kappa_prev
```

**fp32 kappa round-trip (CPU parity).** The CPU carries `kappa` in an fp32 array
between `run_iter` calls and resets (`pipeline.py:365, 513`; `session.py:559, 620`).
So: the seed of the first `iter_m` of every `run_iter` is
`kappa_prev = f64(f32(kappa_final_of_previous_run_iter))`, and after `reset_intra`
it is `f64(f32(ini_val))`; the un-rounded fp64 `kappa_final` is what Phase D uses
within the same `run_iter` (`kappa_f32 = f32(kappa_final)`, `cdln_val`).

One D2H per `iter_m` carrying `(all_flag, drift, kappa)` from a pinned slot; no
host `invad`. `cdln_val = f32(Cdln(kappa_final, dim/2 − 1))` is computed by a
separate 1-thread launch after the loop exits (not per `iter_m`). Device `invad`
costs 2.5 µs (asymptotic branch) / ~140 µs (secant branch, ~20 % of calls on the
bench) single-threaded; acceptable (§8).

### 1.5 Phase C — spatial_connect (gMSHBM) — exact except `u_update`/`cross` order

Per subject (CPU `_kernels.py:432-547`), all `t`-invariant because the gradient
is session-invariant; evaluated on the active support (§2.3):

```
sum_lambda[l] = serial ascending-n fp32 sum over members(l) of s_lambda      (exact; native fp32, §1.10 note)
u_update[d,l] = Σ_{n ∈ members(l)} grad[n,d] · s_lambda[n,l]                  (reassoc; fp64 acc, f32 store)
inv_l         = f32(1) / sum_lambda[l]        0/0 → NaN, must propagate (no guard)
u[l,d]        = u_update[d,l] * inv_l         (fp32 multiply — two roundings, as CPU)
u_sq[l]       = serial ascending-d fp32 sum of u[l,d]²                        (exact given u)
grad_sq[n]    = serial ascending-d fp32 sum of grad[n,d]²                     (exact; once per subject at ctor)
cross[p]      = Σ_d grad[n,d] · u[l,d]        for active p only               (reassoc; fp32 serial over d)
vmf           = (f32(2) * cross - grad_sq[n]) - u_sq[l]                       (left-to-right fp32)
log_connect[p]= T sequential fp32 adds of vmf starting from 0                 (exact; NOT T*vmf)
```

NaN is kept (empty parcel ⇒ whole same-hemi column NaN ⇒ Phase D dead-column
zero). Within P there are no cross-hemisphere cells (block-diagonal bm), so the
P-layout kernels need no `-inf` fill; the CPU's dense `log_connect` **is** `-inf`
on cross-hemisphere cells and K7 must reproduce that (§3 K7).

### 1.6 Phase D — E-step

```
lv_sum[p]   = Σ_t inv[t,n] · ( Σ_{d: bit(t,n,d)} nu[t,l,d]  −  mean[t,n] · S[t,l] ),  S[t,l] = Σ_d nu[t,l,d] (fp64)
              (inner bit-sum fp32 serial ascending d; combine in fp64; f32 store)     reassoc; active p only
log_vmf[p]  = kappa_f32 * lv_sum[p] + f32(n_alive[n]) * cdln_val         (= CPU's kappa_f32*lv + cdln_add; no FMA)
tmp_idx[n]  = (n_alive[n] == 0)
lam[p]      = log_vmf[p] + log_theta[p]    if theta_bits[p] != 0 else -inf        (log_theta = f32(log(f64 theta)), §1.10)
lam[p]     += beta_f32 * log_connect[p]    (gMSHBM)  (fp32 mul, fp32 add)
rmax[n]     = max over l ∈ R(n) of lam, skipping NaN; if the result is NaN or ±inf → 0
scr[p]      = exp( f64(lam[p]) − f64(rmax[n]) )       fp64, stored fp64   (device exp: ≤ 1 ulp vs x86 libm — deviation)
dead col l  : cs = fp64 sum over active members(l) of scr skipping NaN; cs == 0 → scr[members(l)] = 0   (order-free decision)
```

**The set R(n) for rmax.** The CPU takes the max over all `L`. Cells outside P have
`lam = log_vmf + log(theta_out) [+ beta·lc_dense]` where, for gMSHBM, `lc_dense`
is `-inf` on cross-hemisphere cells (the CPU pre-fills them) and the vmf
expression on same-hemisphere cells; for dMSHBM there is no `lc` term at all.
Such cells are `-inf` whenever `theta_out == 0`, i.e. from the first E.2 on. So:

* while `theta_out != 0` (only the very first `run_iter`): `rmax = max(rmax_P(n),
  rmax_out(n))` where `rmax_out(n) = max over l ∉ P(n) (same hemisphere only for
  gMSHBM)` of the dense expression, computed by the **iteration-1 dense pass**
  (§3 K7) with the same fp32 op order;
* afterwards `rmax = rmax_P(n)` — bit-identical to the dense CPU rmax.

Cost (fp32, per row, ascending `l` over P(n); CPU `_kernels.py:742-805`):

```
row_sum = serial fp32 Σ_l f32(scr[p]) * 1.0f
slc     = row_sum > 0 ? f32(scr[p]) / row_sum : 0 ;  NaN → 0 ;  tmp_idx[n] → 0
ltheta  = theta_bits != 0 ? log_theta[p] : LOG_EPS20 ;  ±inf → LOG_EPS20
lslc    = slc   > 0 ? f32(log(f64 slc))   : LOG_EPS20 ;  ±inf → LOG_EPS20
c += slc * log_vmf ;  c += slc * ltheta ;  c -= slc * lslc
lc = log_connect[p]; NaN or ±inf → LOG_EPS20 ;  c += (beta_f32 * slc) * lc      (gMSHBM)
```

`LOG_EPS20 = f32(log(eps_f64**20)) = -720.87305`. Per-row costs are summed in
fp64 with a fixed tree (deviation from the CPU's serial fp32 sum over N; it only
feeds the 1e-4 rel-diff test). The cost chain is native fp32 (FTZ) — subnormal
`slc` terms are lost; their contribution is < 1e-35 of the cost (deviation).
Inactive cells (`theta_bits == 0`) contribute `slc = 0` exactly; their `log_vmf`
is stale but finite (buffers are zero-initialised and only ever written finite).

**K8 `lam` / `-inf` invariant (load-bearing).** At `theta == 0` the device writes
`lam = -inf` (no `lc` term) so that `scr = 0` rather than the CPU's NaN; this is
required because K9's NaN sweep only covers the active CSC. The two agree because
the CPU's dead-column pass zeroes any all-NaN column. Concretely: the CPU adds
`beta·log_connect` unconditionally after the `theta > 0` branch, so at an inactive
cell with a NaN `log_connect` it produces `-inf + beta·NaN = NaN` (hence
`scr = NaN`), and that NaN is swept by the CPU's dense dead-column pass. The
device folds the `lc` add *inside* the branch, giving `-inf` and `scr = 0`, which
is what keeps a NaN out of `estep_row2`'s `rs` accumulation — an all-NaN `rs`
would zero the entire row's `s_lambda`. **Do not align K8 to the CPU's
unconditional `+ beta·lc` without widening K9's sweep to the full `col_ptr` CSC in
the same change**, and vice versa: the two are only correct as a pair.

### 1.7 Phase E — exact (with §1.10 stores)

```
E.1 per row n, ascending l over P(n):  rs = fp64 serial Σ scr[p]  (bm ≡ 1 on P)
    out[p] = rs > 0 ? f32_rn(scr[p] / rs) : 0 ;  tmp_idx[n] → out[p] = 0      (f32_rn: subnormal-aware)
E.2 theta[p] = f32_rn( f32_rn(...f32_rn(0 + sl[0,p]) + sl[1,p] ...) * inv_S_f32 )   — the CPU's fp32 serial
    sum over s and the fp32 scale, emulated in fp64 with per-op rounding (§1.10);   theta_out = 0
    log_theta[p] = theta_bits != 0 ? f32(log(f32_to_f64(theta[p]))) : -inf
```

### 1.8 Outer leaves — exact except device libm

`intra_subject_var_loop` (L17), `intra_em_cost_step2` (L16), `inter_subject_var`
(L18) and the two resets move on-device, mirroring `step2_em_outer/_kernels.py`
op-for-op (R3 §2.2-2.5): fp32 `nu_sum` over `t` (start 0.0f), fp32
`sigma_l*nu_sum + eps_mu` (no FMA), fp64 norm², **`nrm == 0 ⇒ psi_new[s,l,:] = 0.0f
and cos = 0.0f`** (NOT the M-step's inf/NaN convention), else `f32(1.0/nrm)`,
fp32 cosine serial over d, `flag_psi` **latched** across iterations;
`sigma_in = acc * (1.0/(double)(S*T))` (multiply by a precomputed reciprocal, not a
divide) → `min(·,1)` → `invad` → `max(·, ini_val)` → `isfinite` fallback to the
previous value; `rel_mean = Σ_l |f64(f32(sig_cur − sig_new)) / f64(sig_cur)| / L`
(fp32 subtract first); the loop runs at most `max_iter_intra_var` iterations with
the test at the end. `mu` update: fp64 sum over `s`, fp32 store, norm² of the
**unrounded** fp64 sum, fp32 scale, zero-norm fallback to `prev_mu`;
`eps_in = acc * (1.0/(double)S)`. `intra_em_cost`: fp64 per-`l` dots +
`Cdln(sigma_l)`, `Cdln(epsil_l)`, epilogue in ascending `l`.

### 1.9 Bit-exact vs reassoc vs deviation, summarised

Exact: row stats/X, init, sigma_psi, sum_lambda, u/u_sq/grad_sq/vmf/T-fold given
`u_update`/`cross`, lam assembly given `lv_sum`, rmax from iteration 2, the
dead-column decision, E.1 and E.2 given `scr`, per-row cost given `scr`,
outer-leaf arithmetic given the libm scalars. Reassoc: `X_dot_sl`, `lv_sum`, `u_update`, `cross`,
`kappa_sum`, `denom`, cost total. Deviations (§8): M-step norm/cos fp64, device
`exp` (≤ 1 ulp), device `Cdln`, device `invad` (≤ 1e-12 rel, which is the
tolerance `test_device_invad_matches_invad_numba` pins — not bit-exactness;
~3e-4 on the rare secant early exit; invisible after the fp32 kappa
round-trip), FTZ on the cost chain and on `w_n`/`sum_lambda`.

### 1.10 Flush-to-zero: what CuPy forces and how we stay exact

CuPy appends `-ftz=true` after every user option (`cupy/cuda/compiler.py:553`),
so **every fp32 arithmetic op, compare and conversion in the RawModule flushes
fp32 subnormals**; fp64 is unaffected. The CPU keeps subnormal fp32 `s_lambda`
and `theta` cells (values in [1.4e-45, 1.18e-38], i.e. `lam − rmax ∈ [−103.3,
−87.5]`), and `theta = 0` is absorbing, so those cells must survive on device.
Rules:

* `__device__ float f32_rn(double x)`: if `|x| ≥ 2^-126` return `(float)x`;
  else `m = rint(x * 2^149)` (fp64, RNE, exact scaling) and return
  `__int_as_float(sign | (int)m)` (subnormal bit pattern; `m ≤ 2^23`).
* `__device__ double f32_to_f64(float f)`: `b = __float_as_int(f)`; if
  `(b & 0x7f800000) == 0` return `±(b & 0x7fffff) * 2^-149` in fp64, else `(double)f`.
* Emulated fp32 ops: `f32_rn((double)a op (double)b)` reproduces a native fp32
  op with subnormal support (fp64 has ≥ 2·24+2 bits, so double rounding is
  innocuous).
* Tests on possibly-subnormal fp32 values use bit patterns: `theta > 0` ⇒
  `__float_as_int(theta) > 0` (positive non-zero, incl. subnormal);
  `s_lambda != 0` ⇒ `(bits & 0x7fffffff) != 0`.
* Sites that MUST use these helpers: E.1 store, E.2 accumulate + scale,
  `log_theta` (widen via `f32_to_f64`), every `theta`/`s_lambda` non-zero test,
  the active-support compaction flag, `theta_export`. Sites left native
  (deviation, negligible): the cost chain, `w_n = s_lambda·inv` in K3,
  `sum_lambda` in K5a.
* A unit test feeds a row whose `scr/rs` lands at 1e-40 and asserts the stored
  `s_lambda` and `theta` bit patterns equal numpy's.

## 2. Device state and layout

### 2.1 Static P-layout (`arealmshbm/step2_io/sparse_layout.py`)

```python
@dataclass(frozen=True)
class Step2Layout:
    N: int; L: int; n_lh: int; L_lh: int; P: int
    row_ptr:  np.ndarray   # (N+1,) int32, CSR over ALL N rows (empty rows allowed); cols ascending within a row
    col:      np.ndarray   # (P,)   int32 parcel index
    p_row:    np.ndarray   # (P,)   int32 row id of CSR position p
    col_ptr:  np.ndarray   # (L+1,) int32, CSC; members(l) = csc_row[col_ptr[l]:col_ptr[l+1]] ascending n
    csc_row:  np.ndarray   # (P,)   int32 vertex id
    csc_pidx: np.ndarray   # (P,)   int32 index into CSR order
```

Invariants (raise `ValueError`): `N == 2*n_lh`, `L == 2*L_lh`, no cross-hemisphere
cell, every value exactly 1.0 after `eliminate_zeros`, `P > 0`. Builders:
`build_step2_layout(lh_mask, rh_mask)` (any scipy sparse, converted to CSR
internally) and `build_step2_layout_dense(bm_NL)` (reference for tests);
`layouts_equal(a, b)`; `layout_to_device(layout) -> dict`.

### 2.2 Device buffers (S subjects)

| name | shape / dtype | notes |
|---|---|---|
| `packed` | `(S, T, N, Db)` uint8 (`eager_bitpacked`) or one `(T, N, Db)` slot (`stream`) | on-disk layout; padding bits zero (writer contract; cheap check on subject 1) |
| `row_mean`, `row_inv`, `n_alive` | `(S, T, N)` fp32 ×2, `(S, N)` int32 — device-resident for ALL S in both cache modes (computed once in the ctor; stream mode only H2Ds the packed bytes per visit) | §1.1 |
| `n_alive` | `(S, N)` int32 | |
| `grad`, `grad_sq` | `(S, N, Dg)` fp32, `(S, N)` fp32 | gMSHBM only; device-resident (no stream mode for grad in v1) |
| `mtc_DL`, `mtc_LD` | `(D, L)`, `(L, D)` fp32 | init argmax / reset source |
| `mu`, `s_psi`, `sigma_psi` | `(L, D)`, `(S, L, D)`, `(S, L, D)` fp32 | |
| `s_t_nu`, `X_dot_sl` | `(S, T, L, D)` fp32 | in-place M-step state |
| `s_lambda` | `(S, P)` fp32 | CSR order; subnormals allowed |
| `theta`, `log_theta` | `(P,)` fp32 | `log_theta` refreshed after every E.2 (fp64 log on device) |
| `theta_out` | host float | `f32(eps)` until the first E.2, then 0 |
| active-support arrays | §2.3 | rebuilt after every E.2 |
| `scr` | `(P,)` fp64 | softmax scratch (per subject, reused) |
| `lv_sum`, `log_vmf`, `log_connect`, `rmax`, `rmax_out` | `(P,)` fp32 ×3, `(N,)` fp32 ×2 | per subject, reused; zero-initialised |
| `sigma`, `epsil`, `sum_lambda`, `u_sq`, `cdln_*` | `(L,)` | |
| `u_update` / `u` | `(L, Dg)` fp32 | |
| `cost_S` | `(S,)` fp64 + pinned host mirror | |
| M-step scratch | `partials (1024,) fp64`, `cos (S,T,L) fp32`, `flag (S,T) int8`, pinned 4-double slot | |
| iteration-1 scratch (freed after) | `X_dense (N, T·D)` fp32 770 MB, `lv_dense (N, L)` fp32, `cross_dense (N, L)` fp32 | §3 K7 |

Sizing rule for `bold_cache_mode='auto'` (device bytes, evaluated after draining the cupy pools so `mem_info` reflects what this process can allocate): `packed` + `S·P·4` (`s_lambda`) + `2·S·L·D·4` (psi ping-pong) + `2·S·T·N·4` (row stats) + `S·N·4` (`n_alive`) +
`grad`/`grad_sq` + `2·S·T·L·D·4` (`s_t_nu`, `X_dot_sl`) + `2·S·L·D·4` + the 1 GB
iteration-1 scratch + `bold_cache_safety_margin_gb` must fit in free memory,
else `stream`. An explicit `eager_bitpacked` that does not fit raises
`ValueError` with the sizes (the old backend's contract); `auto` warns and falls
back. an explicit `stream` still validates that the resident state fits (named `ValueError` otherwise). `stream` = per-visit H2D of one subject's packed bytes (no row-stats recompute) from a **host**
packed cache `(S, T, N, Db)` uint8 (pageable numpy, decoded once in
`load_inputs`; S·72 MB host RAM — document it) into the pinned slot; two visits
per subject per EM iter (X_dot_sl needs every subject before the M-step; K6 needs
the post-M-step `s_t_nu`). If `grad` alone does not fit, raise (S > ~150 on 24 GB).

### 2.3 Active support (dynamic, exact)

After init and after every E.2 build, from `active[p] = (theta_bits[p] != 0)`:

* `act_p (P_act,) int32` — active CSR positions in ascending order (exclusive scan
  of the flag over p; `cupy.cumsum` is a deterministic scan);
* `act_col_ptr (L+1,)`, `act_csc_row`, `act_csc_pidx` — the CSC restricted to
  active cells (scan of the flag in CSC order, gathered through `csc_pidx`).

**The flag is the union**, not `theta` alone: `active[p] = (theta_bits[p] != 0)
|| (∃ s: s_lambda[s, p] != 0)`. `theta` alone is not sufficient because E.2 rounds
(`th = f32_mul_rn(acc, inv_S_f32)`): at `S ≥ 2` a cell whose only nonzero
`s_lambda` is the minimum subnormal `2^-149` gives `acc · f32(1/2) = 2^-150`, an
exact tie that rounds half-to-even to **0**, so `theta == 0` while
`s_lambda != 0` and K3/K5a would skip a member the CPU visits. Measured on the
S=2 bench: 158 such cells after EM iter 1, then 44 / 19 / 12 / 7 / 4 over iters
2-6, every value exactly `1.401298e-45`. The union costs nothing —
`theta_mean_logtheta` already loops over `s` to build `acc`.

Correctness: an inactive cell has `theta = 0 ⇒ lam = -inf ⇒ scr = 0 ⇒ s_lambda = 0`
for every subject, so skipping it in K3 (`fmaf(0,·,a) == a`, `B += 0`), K5a
(`+0.0`), K5b/K6 (never read), K9 (`+0.0`) and the row kernels' reads is
bit-identical to visiting it. `supp(s_lambda[s]) ⊆ supp(theta)` holds from the
second EM iter; at iteration 1 `theta = eps > 0` on all of P so the full P is
visited exactly once. K8/K10 still walk each row's full P(n) (≤ 15) and write
explicit zeros for inactive cells.

## 3. Kernel catalog

One `cupy.RawModule` (`-std=c++17 -fmad=false`, lazily compiled, process-global)
in `arealmshbm/step2_em_iter_master/_kernels_gpu.py`. Rules: no
floating-point atomics; every reduction is a fixed-shape tree over a fixed grid;
kernels never allocate; `long long` indexing where products can exceed 2^31;
each kernel has a thin host wrapper validating shapes/dtypes/contiguity. Measured
host floors on this box: 6.1 µs per launch, 15 µs per null-stream sync, 44 µs per
pinned-slot D2H (`.get(out=pinned)`).

| # | kernel | geometry | contract |
|---|---|---|---|
| K0 | `row_stats_exact` | thread per (s,t,n) row | §1.1; writes `row_mean`, `row_inv`; a second kernel counts `n_alive` |
| K1a | `init_hard_labels` | block per 8 rows (s,n): 256 threads; lanes/warps own `l`; `g[d, :]` staged per d-tile and reused across the 8 rows | §1.2; per-(n,l) accumulator stays a private serial ascending-d fp32 sum; warp argmax with first-index tie rule; `norm_sq` serial fp32 |
| K1b | `init_compose` | thread per (s,n); then thread per p | §1.2: binary-search `l_act` in `col[row_ptr[n]:row_ptr[n+1]]`; `theta[p]` from `Σ_s s_lambda[s,p] > 0` |
| K2 | `sigma_psi_SLD` | grid-stride | exact |
| K3 | `x_dot_sl_bits` | block per (s,t,l), 128 thr, dyn smem `64*4 + 64*Db` (≤ 17.7 KiB) | verbatim step-3 port over the **active** CSC; wrapper requires `Db ≤ 256` (raise `ValueError` naming the seed mesh: `backend='gpu'` supports `ceil(D/8) ≤ 256`, i.e. seed mesh fsaverage3) |
| K4a | `kappa_sum_stage1` / `reduce_stage2` | fixed grid 1024×256 / 1 block | fp32 product, fp64 tree |
| K4b | `mstep_flags_kappa` | (S·T) blocks for the flags; the last block (or a 1-thread tail) computes `rbar`, device `invad`, clamps, `drift`, `kappa_f32`, `all_flag` → pinned slot | one launch fewer than a separate kappa kernel; the flag pass and the scalar epilogue are ordered by launching the epilogue as a 1-block kernel after the flags (2 launches) if a single-launch grid-wide ordering is not available — either way ≤ 6 launches per iter_m |
| K4c | `mstep_fused_body` | block per (s,t,l), 256 thr, dyn smem `4*D` | §1.4; `kappa_f32` read from the device slot; **read `old` into a register before the in-place store**; writes `cos[s,t,l]` |
| K4e | `cdln_after_loop` | 1 thread | `cdln_val = f32(Cdln(kappa_final))`, once per `run_iter` |
| K5a | `connect_u` | grid (L, ntile=4): member tiles, fixed-tree combine in a second stage | `sum_lambda` serial fp32 (ascending members, one thread), `u_update` fp64, `inv_l`, `u`, `u_sq` serial fp32 |
| K5b | `connect_scv_P` | thread per active p | `cross` fp32 serial over d; `vmf`; T sequential adds; NaN kept |
| K6 | `acc_P` (lv_sum on active P) | block per (s,l,tile): 256 thr; members of the active column in chunks of 256; inner loop over t restages `nu[s,t,l,:]` **as `Db*8` floats with the tail zero-filled** + `S[t,l]` in smem | thread per member: 147 packed bytes (`__ldg`), `pt = serial ascending-d fp32 Σ` via branch-free `fmaf((float)bit, nu_sm[d], pt)` (measured bit-identical to set-bit iteration and 1.9× faster), `part += (double)inv * ((double)pt − (double)mean * S)`; after T: `lv_sum[p] = f32(part)`. `S[t,l]` from a block-per-(s,t,l) fp64 row-sum kernel. Measured 3.07 ms on full P → ~0.4 ms on the active support |
| K7 | iteration-1 dense pass (only while `theta_out != 0`) | (i) `widen_exact`: `X_dense[n, t·D+d] = (f32(bit) − mean)·inv` (thread per row; exact); (ii) `lv_dense = X_dense @ nu.reshape(T·D, L)` via cuBLAS sgemm (**16.6 ms**; `nu` transposed to `(T,D,L)` first); (iii) `cross_dense = grad @ u.T` (sgemm, 0.3 ms); (iv) `rmax_out_k`: warp per (s,n), lanes over l ∉ P(n) (smem scatter of the row's P cols), gMSHBM: same-hemisphere only, `lc = T-fold of ((2f·cross − grad_sq) − u_sq[l])`; dMSHBM: all l, no lc term; `lam = (kappa_f32·lv + f32(n_alive)·cdln) + f32(log(theta_out)) [+ beta_f32·lc]`; NaN skipped; `-inf` if none | ~35-40 ms once per run (S=1), 1 GB transient freed after. No TF32 scope wraps step 2 (the dense port's `enable_tf32` knob went with it), so the sgemm runs strict fp32 |
| K8 | `estep_row1` | thread per row n (measured: better than warp-per-row at ≤ 15 cells) | §1.6 up to `scr`; reads `rmax_out` when the flag is set; skips `lv_sum` reads for inactive cells; writes `log_vmf[p]`, `scr[p]`, `rmax[n]` |
| K9 | `dead_col` | grid (L, ntile) over active members + fixed-tree combine | fp64 NaN-skipping sum; if 0 → write 0 to those `scr[p]` |
| K10 | `estep_row2` | thread per row n | cost row (fp32) → block partials fp64; E.1 (§1.7, `f32_rn` store) → `s_lambda[s,p]`; `tmp_idx` from `n_alive`; explicit zeros for inactive cells |
| K11 | `reduce_stage2` | 1 block | `cost_S[s]` fp64 |
| K12 | `theta_mean_logtheta` + `build_active` | thread per p; scans via cupy | §1.7 E.2 with `f32_rn` emulation; `log_theta`; then §2.3 |
| K13a | `intra_psi_iter` | block per (s,l), threads over d | §1.8 one iteration; zero-norm branch |
| K13b | `intra_sigma` | block per l (fp64 dot over (s,t,d), fixed tree) then **300 blocks × 1 thread** for `invad` (measured 2.4× faster than one 300-thread block) | `sigma_new`, flags latch (per-s count over l), `rel_mean` → 2-double pinned slot |
| K13c | `inter_mu`, `inter_eps_dot`, `inter_epsil` | block per l (fp64 sum over s, serial fp64 norm²); **`inter_eps_dot` is deliberately a serial fp64 dot on one thread per parcel** (CPU order) — a fixed tree changes `R` by ~1 ulp and `epsil = invAd(dim, R)` with `R → 1` amplifies that by ~5e9, which at S=1 moved the outer stop test from inter 10 to 7 and cost 4 theta flips; it costs 0.16 ms × 10 per run; `invad_L` as L blocks × 1 thread | §1.8 |
| K13d | `intra_em_cost` | block per l + reduce; `Cdln` as L blocks × 1 thread | §1.8 |
| K13e | `reset_broadcast` | grid-stride | `s_t_nu ← mtc_LD`, `s_psi ← mtc_LD`, `sigma ← ini_val` |
| K14 | `theta_export` | — | `theta` D2H → host csc from `(p_row, col, val)` |

### 3.1 Device scalar functions (fp64)

Port `em_stop_criterion/_cdln.py::_log_bessel_i_debye` (5-term Debye) and
`_cdln_single`; `_log_bessel_i` = Debye for `v ≥ 25`, else return NaN (the
series branch is unreachable at `dim = 1174`; a future small `dim` must fail
loudly, and the host wrapper asserts `dim/2 − 1 ≥ 25`). Port
`m_step/_invad.py::invad_numba` verbatim (Banerjee init, probe
`log I_v(k0) > 709 or < -708 or NaN → asymp`, secant ≤ 50 iters with `|df| < 1e-300`
and `xtol 1e-12` exits, `rbar ≤ 0 → 0`, `rbar ≥ 1 → inf`) — the device port
reproduces `invad_numba(1174, rbar)` bit-exactly on the asymptotic branch; on the secant branch ~12 % of `rbar` values differ by ≤ 1e-12 and ~1 in 4000 hits a secant early-exit divergence of ~3e-4 relative (device vs x86 libm `log`/`exp` ULPs compounding through the iteration) — see §8.
`initialize_concentration` stays host (scipy, once).

## 4. Session (`arealmshbm/step2_em_iter_master/session_gpu.py`)

```python
class Step2SparseSession:
    def __init__(self, inputs: "Step2SparseInputs", *, mode, num_clusters, dim, ini_val,
                 beta_internal, eps_m_step, max_iter_m, eps_intra_var, max_iter_intra_var,
                 bold_cache_mode="auto", bold_cache_safety_margin_gb=4.0)
    # ctor: layout → device; sizing rule (§2.2); packed BOLD → device cache (filled through
    #       inputs.bold_reader into ONE pinned (T,N,Db) slot, subject by subject) or stream slot;
    #       K0 row stats; grad + grad_sq (via one pinned (N,Dg) slot); mtc; allocate every
    #       buffer (zero-filled where §1.2 says so); compile the module. No EM work.
    S: int                                    # subject count (seeds the shared EM loop's cost vector)
    def initialize_state(self) -> None        # K1 + log_theta + active support; theta_out = f32(eps);
                                              # sigma/epsil = ini_val; kappa seed = f64(f32(ini_val)); mu/s_psi/s_t_nu ← mtc
    def reset_inter(self) -> None             # sigma[:] = ini_val ; s_psi ← mtc_LD
    def reset_intra(self) -> None             # kappa seed = f64(f32(ini_val)) ; s_t_nu ← mtc_LD
    def run_iter(self) -> Tuple[int, np.ndarray]   # one outer EM iter; (m_iters, cost_S host fp64 copy)
    def intra_closure(self) -> float          # L17 then L16 → update_cost (host float)
    def inter_closure(self) -> None           # L18 on device
    @property kappa(self) -> float            # last kappa_final (fp64) for logging / Params['kappa']
    def export_params(self) -> Dict[str, Any] # host: sigma/epsil/kappa (1,L) f32, mu (L,D) f32, cost_em (S,) f64,
                                              # theta (scipy.sparse.csc_matrix (N,L) fp64), ini_val.
                                              # The theta key is exactly 'theta' — there is no 'theta_csc'
                                              # spelling anywhere; `_run_em_sparse` reads `exported['theta']`.
    def sync_to_host(self, fields) -> Dict    # tests/A-B: 's_t_nu' (S,T,L,D), 's_lambda' dense (S,N,L), 'theta' dense (N,L),
                                              # 's_psi', 'sigma', 'epsil', 'mu', 'lv_sum', 'log_connect', 'scr', 'rmax' (per-subject dense)
    timings: Dict[str, float]                 # host wall, seconds, cumulative: session_ctor,
                                              # initialize_state, intra_closure, inter_closure
```

`run_iter` sequence (S subjects): K2 → [per s: BOLD to device if stream; K3] →
denom → M-step loop (K4a, K4c, K4b; one D2H per iter_m) → K4e → S-row-sum →
[per s: BOLD if stream; K5a, K5b (gMSHBM); K6; K7 (iteration 1 only); K8; K9;
K10; K11] → K12 → `cost_S` D2H. Blocking syncs per `run_iter` = `m_iters + 1`;
launches ≈ 14 + 5·m_iters + 9·S (≈ 0.25 ms of host floor at m_iters = 2.5, S=1).

The EM body loop (`vmf_clustering_batch` semantics: `rel = |(cost−prev)/prev|`
per subject, all subjects must pass, `iter_em == 1` never stops, cap
`max_iter_em`) lives in `step2_pipeline/vmf_clustering_batch.py::em_body_sparse`.

## 5. Inputs (`arealmshbm/step2_io/sparse_inputs.py`, `arealmshbm/data_io/mat5_stream.py`)

```python
@dataclass
class Step2SparseInputs:
    layout: Step2Layout
    S: int; T: int; N: int; D: int; D_grad: int; n_lh: int; n_rh: int
    mtc: np.ndarray                     # (D, L) fp64, verbatim group.mat 'mtc' (D = 1175 here; dim = D-1)
    dim: int
    bold_reader: Callable[[int, np.ndarray], None]   # bold_reader(s_1, out_TNDb): fill a caller buffer (any mode)
    packed_host: Optional[np.ndarray]   # (S, T, N, Db) uint8, PAGEABLE, present when the host cache was built
                                        # (default for every mode except when S·T·N·Db exceeds the host budget)
    grad_reader: Optional[Callable[[int, np.ndarray], None]]   # grad_reader(s_1, out_NDg) (gMSHBM), else None
    timings: Dict[str, float]
    @classmethod
    def from_arrays(cls, layout, packed_STNDb, mtc, *, D=None, grad_SNDg=None) -> "Step2SparseInputs"  # tests
                                        # NOTE the keyword is ``D`` (the profile cell count), NOT ``dim`` (= D-1).

def load_step2_sparse_inputs(cfg: Step2Config, *, overlap: bool = True) -> Step2SparseInputs
```

* cohort.json + the same consistency checks as today (`num_sub`, `num_session`,
  mesh targ/seed); MW contract verification of subject 1 / session 1 (packed
  bytes at MARS rows must be zero); dims from vlmeta.
* BOLD: `blosc2.open(path, mode='r')`, `a[:]` (≈ 45 ms per subject, GIL-released)
  on a ≤ 4-worker pool into the pageable host cache; `bold_reader` copies from the
  cache when present, else decodes from disk. Host budget: build the cache when
  `S·T·N·Db ≤ 0.5 × available RAM` (psutil if present, else 16 GB), else
  `packed_host = None` and stream mode decodes per visit (document the 2·S·45 ms
  per EM iter bill).
* gradient: suffix-sniffed like `SubjectGradientLoader` (`.npy` → `np.load[:, :Dg]`,
  `.mat` → `load_subject_gradient._read_emb_100`), concatenated to `(N, Dg)` fp32,
  no T replication.
* `group.mat`: `mat5_stream.read_fields(path, {'mtc'})` — streaming inflate
  (isal when installed, stdlib `zlib` otherwise) that never inflates the 22 MB
  `lambda` field's body: the container cursor advances by the raw element size,
  and a skipped compressed element is inflated only far enough to read its name.
  Header walk confirms `epsil`, `lh_labels`, `rh_labels` exist. Falls back to
  `scipy.io.loadmat` on any `_Unsupported`. Values bit-identical to scipy.
* spatial mask: the `.mat` stores **MATLAB sparse** (`mxSPARSE_CLASS = 5`,
  fields `ir` (int32 rows), `jc` (int32 indptr), `pr` (fp64)). `mat5_stream.read_sparse(path, name)`
  parses them straight into a scipy `csc_matrix` (no nonzero scan); dense-class
  branch kept for hand-made files; scipy fallback. Then `build_step2_layout`.
* Meshes: `load_avg_mesh` (cached npz) only for `n_lh`, `n_rh`, MARS.
* The four reads overlap on a 4-worker pool when `overlap=True`; per-item timings
  (`cohort`, `mesh`, `gradients`, `group_mtc`, `profiles`, `boundary`, `total`).
  The pipeline forwards those under `load_inputs.<item>` **except `total`**,
  which it drops — `Step2Result.timings` has no `load_inputs.total`; the
  equivalent number is `load_inputs`, timed by the pipeline around the call.

`mat5_stream.py` is the generic MAT-v5 walker (`_InflateReader`, tag/element
parsing) with `read_fields`, `read_sparse`, `walk_names` entry points and a
`LAST_PATH` diagnostic; synthetic-file tests compare against scipy (dense,
sparse, compressed and uncompressed containers). Step 3's
`step3_pipeline/sparse_inputs.py` reuses these primitives (plus
`_InflateReader.iter_raw`) to stream the prior's theta straight into a CSR
build rather than materialising the matrix this module returns.

## 6. Pipeline, config, save, driver

* `Step2Config.backend ∈ {'cpu', 'gpu'}`; `'gpu'` = this backend.
  `PipelineConfig.backend_step2` takes the same two values as step0/step1
  (`_VALID_BACKENDS_STEP012`). `pipeline.__exit__` frees the pools on `'gpu'`.
  There is no step-2 TF32 scope or `enable_tf32` knob (see the K7 note).
* `Step2Pipeline.run()` dispatches: `'gpu'` → `load_inputs_sparse()` →
  `_run_em_sparse()`; the CPU path is untouched. `Step2Inputs` and
  `Step2SparseInputs` are both re-exported from `step2_pipeline/__init__.py`.
* `_run_em_sparse`: same outer/intra/EM control flow, same log lines, same
  progress emits, same `Record`/convergence arithmetic as `run_em`, calling the
  §4 API. `Step2Result.timings` gains `init_device`, `closure_total`, `save`.
* Save: `_save_params` moves to `step2_pipeline/_save.py::save_params_final(Params,
  path)` (T2 lands this first; `pipeline.py` keeps a one-line delegation). `mu`
  is transposed to `(D, L)` as before, and the file is the legacy dense fp64
  `theta` block with `do_compression=True` on every backend — the sparse
  backend's `export_params()` hands over a `csc_matrix` and the writer
  densifies it, because CBIG's MATLAB `log(Params.theta)` does not accept a
  sparse input. `data_io/load_group_prior.py` densifies a sparse `theta`
  (`hasattr(x, 'toarray')`) anyway and keeps returning `(N, L)` fp32, so a
  MATLAB-side `sparse()` prior is still a valid external input.
* Driver: `arealmshbm/pipeline/step2_runners.py::prewarm_step2_gpu(background=True)`
  (module-global daemon thread + lock, swallow-all, idempotent) runs
  `warmup_step2_gpu()` (no pinned buffers: those are allocated at
  Session-ctor time — the driver drains the pools after step 0); called from
  `driver.run()` right before step 0 when Mode B and
  `backend_step2 == 'gpu'`. `Pipeline.__init__`
  (`_validate_inputs_vs_config` → `_require_fsaverage3_seed`) rejects a
  `seed_mesh` other than `fsaverage3` for `backend_step2` (Mode B only) and
  for `backend_step3`, so a cohort that the sparse kernels cannot take does
  not get discovered after step 1 (the other two static limits,
  `num_clusters <= 32*INIT_MAXJ` = 512 and
  `n_grad_components <= (49152 - 2064)/4` = 11772 — `connect_u`'s dynamic
  shared memory against the 48 KiB per-block default — are refused by the
  config parser and by `Step2Config.__post_init__`, both mirroring
  `_kernels_gpu`'s `MAX_CLUSTERS` / `MAX_D_GRAD` as literals so the
  stdlib-only parser never imports cupy).
  `warmup_step2_gpu()` also runs a 32x32 gemm, which pays cuBLAS's
  process-wide kernel-module load that the iteration-1 dense pass (K7) would
  otherwise charge to the first `run_iter`. cupy's cuBLAS handle is per
  **thread**, so this cannot disturb the handle step 0 is using on the main
  thread. `_run_em_sparse` calls it inline as well, so the per-step API and the
  profiling harnesses (which never prewarm) do not pay it inside `em_total`.
* `arealmshbm/__init__.py` → PEP-562 lazy re-exports; `step2_pipeline/pipeline.py`
  imports the CPU session / outer leaves lazily inside the CPU branch so a
  GPU run never imports `step2_em_outer` (eager `_cdln` numba signatures, ~0.2 s).
* Harness: `arealmshbm/step2_pipeline/profile.py` — `--project`, `--backend`,
  `--runs`, `--prewarm`, `--num-sub/--num-session/--num-clusters`, `--beta`;
  prints the per-stage table (load, init, ctor, em_total with per-iter mean and
  iter count, closure, save, total) plus the Session's cumulative host walls.
  Stage timing only — the per-kernel cuda-event timers this Session once
  carried were removed (a host-blocking `synchronize()` per timed group, worth
  ~6-10 % of `em_total`, in production code); use Nsight for a per-kernel
  breakdown.
* A/B: an internal step-2 comparison harness — runs `cpu`/`gpu`
  (or loads existing `Params_Final.mat`), densifies theta through a
  `_densify` shim before any comparison, reports per-inter `Record`, `kappa`,
  `sigma`, `epsil`, `mu` deltas, theta argmax flips / support-diff rows /
  dead-alive rows (alive from the packed BOLD), EM iter counts, and run-to-run
  identity on `(theta.indptr, theta.indices, theta.data)` + every other field.
* Docs: this file; `docs/step2_flow_and_subgraphs.md` "Backends" + wall table;
  `docs/step2_em_iter_master_kernel.md` corrections (R1 §13: LOG_EPS20 value,
  fp32 GPU-dense caveats, `(N,T,D)` layout, retired E-step paragraph);
  `lib/hyperparameters/step2.json` description text for `backend_step2`.

## 7. Validation

Fixtures: `testdata/step2_bench/proj` (S=1) and `proj2` (S=2), env override
`MSHBM_STEP2_BENCH_DIR`; tests skip when absent or without cupy. CPU references:
`testdata/step2_bench/out_cpu_ref/run0/Params_Final.mat` and `out2_cpu_ref/run0`
(dense theta — the comparison shim densifies both sides).

1. **Kernel unit tests** (`step2_em_iter_master/tests/test_kernels_gpu.py`)
   on the S=1 fixture, each against the CPU numba kernel on identical inputs:
   K0 bit-exact (reconstruct `(bit−mean)·inv` for sampled rows vs
   `_widen_normalize_bitpacked_to_f32_NTD_kernel`, `np.array_equal`); K1
   labels/medial/s_lambda/theta `np.array_equal` vs `compose_init_state`; K3 vs an
   fp64 matmul: max rel ≤ 1e-4 and ≤ 10× the fp32 sgemm's error; K4c in-place: run
   with `eps = 0` for ≥ 2 iter_m, assert `cos != 1` and `s_t_nu` bit-equal to a
   ping-pong reference; K5 `sum_lambda`, `u_sq`, `grad_sq` exact, `log_connect` on
   P within 1e-5 rel of the CPU kernel, NaN pattern equal; K6 vs `X @ s_t_nu` fp64:
   max rel ≤ 3e-4; K7 `rmax_out` vs a dense numpy evaluation of the same expression
   (incl. the cross-hemisphere `-inf` rule); K8-K10 with the CPU
   `_fused_estep_per_subject_NTD` + `_phase_e1_normalize_per_subject_kernel` fed
   the kernels' own `lv_sum`/`log_connect` gathered to dense: `scr` within 4 ulp,
   `s_lambda` within 1 fp32 ulp with identical zero pattern, per-row cost within
   1e-6 rel; the §1.10 subnormal round-trip test; K12 exact; K13 vs the three numba
   leaves: `psi` `np.array_equal`, `sigma/epsil/mu` ≤ 1e-9 rel, cost ≤ 1e-10 rel,
   plus a synthetic all-zero `(s,l)` column; **determinism**: every kernel launched
   twice → equal uint32 views.
2. **Session vs CPU master, 1 EM iter** on a synthetic **binary** cohort built with
   `Step2SparseInputs.from_arrays` (S=2, T=2, `N = 2·n_lh` small, L=16, D ≥ 256 so
   `invad` is asymptotic, real `Step2Layout` from a synthetic block-diagonal mask)
   against `Step2EmIterSession` (CPU) fed the same bits through
   `InMemoryProfileLoader` after the CPU widen: bars `s_t_nu/theta/s_lambda
   ≤ 5e-3`, `kappa ≤ 1e-4` (the retired dense port's parity bars);
   gMSHBM and dMSHBM.
3. **End-to-end A/B** (`compare_step2_backends.py`) on S=1 and S=2: `gpu`
   twice → identical on every saved field (**required**); vs CPU: theta argmax
   flips ≤ 500,
   dead-alive rows within ±100 of the CPU's (1 303 at S=1; the dense port: 4 744 /
   +4 225), `|kappa − kappa_cpu|/kappa_cpu ≤ 2e-2`, first-inter `Record` within
   5e-3 rel.

   **Additionally REPORTED (not barred), every A/B:** `epsil` max-rel *and* the
   conditioning-aware per-parcel `|log10(epsil_a/epsil_b)|` median / max, and
   `mu` per-column cosine min / median. Both are printed by
   `compare_step2_backends.py`. They are excluded from the bars for the reason
   in §8 (`invAd` near `R -> 1` is ill-conditioned; `mu` max-rel measures the
   wrong thing for a unit direction) — but a summary that omits them is
   incomplete, because the headline "matches CPU" is otherwise carried entirely
   by the well-conditioned fields the bar list happens to name.
4. **Full pytest suite** stays green.
5. **Perf** (warm, `profile.py --runs 5`, mean ± span): S=1 steady-state EM
   iter and `Step2Pipeline.run()` wall, with the one-shot iteration-1 K7 pass,
   the fresh-process total and S=2 reported separately. Numbers in §9.

## 8. Deviations and open items

* M-step `‖col‖`/`cos` in fp64 (CPU fp32 serial).
* Device `exp` ≤ 1 ulp (6 % of inputs), device `Cdln`.
* Device `invad`: **≤ 1e-12 rel — that tolerance, not bit-exactness, is what
  `test_device_invad_matches_invad_numba` asserts (`test_device_cdln_bitexact`
  is the bit-exact one); on the secant branch ~12 % of `rbar` values differ by
  ≤ 1e-12 rel and ~1 in 4000 hits a secant early-exit divergence of ~3e-4
  rel** (device vs libm `log`/`exp` ULP
  differences compound through the iteration until one side takes the
  `|x_new − x1| < 1e-12` exit a step earlier). Measured over 4000 `rbar ∈ (0,1)`
  at `dim = 1174`, three seeds: 488-516 differ at all, 0-1 above 1e-12, max rel
  2.98e-4 / 3.35e-4 / 6.4e-13 (worst `rbar = 0.34147`: 453.5874 vs 453.7225);
  band `rbar ∈ [0.1095, 0.5411]` i.e. `κ ∈ [130, 898]`, which straddles the
  production κ range. In the ≤ 1e-12 cases `np.float32(kappa)` is identical on
  both sides, so `kappa_f32`, `cdln_val`, the exported `Params.kappa` and the
  `_kappa_seed` are unaffected and only the fp64 `drift` test inside
  `_mstep_loop` sees it.
* Reassociated: `X_dot_sl`, `lv_sum`, `u_update`, `cross`, `kappa_sum`, `denom`, cost total.
* FTZ left native on the cost chain, `w_n` in K3 and `sum_lambda` (subnormal-only
  effects; a parcel whose every member weight is subnormal would be NaN-dead on
  device and finite on the CPU — astronomically unlikely).
* K3 bit form is ~5× less accurate than the fp32 sgemm at ~230 members/parcel
  (max rel ~6e-5, → δκ/κ ~3e-6); fold `a32` into `acc` every 16 members if a tighter
  band is ever needed.
* Device `invad` secant branch ~140 µs single-thread (≈ 20 % of iter_m on the bench).
* `tmp_idx := (n_alive == 0)`: exact for `κ < 2308`; one-shot warning above.
* K7 uses cuBLAS once per run, in strict fp32 (no TF32 scope wraps step 2).
* `backend='gpu'` requires `ceil(D/8) ≤ 256` (seed mesh fsaverage3); grad must fit
  on device (S ≲ 150 on 24 GB); stream mode costs 2·S·(H2D 72 MB) per EM iter.
* **`epsil` and `mu` are NOT on the §7.3 bar list, deliberately.**
  `epsil = invAd(dim, R)` with `R -> 1` at small `S` is catastrophically
  ill-conditioned, so a ULP-level difference in `R` moves `epsil` by orders of
  magnitude. Measured against the CPU reference: **S=1 `epsil` max-rel 1.0**,
  and this is NOT one bad parcel — **202 of 300 parcels differ by more than
  0.5 relative at S=1** (re-measured 2026-09-04 on a copy of the S=1 bench;
  worst parcel 183: CPU 1.85e10 vs GPU 403.611, i.e. the `ini_val` fallback
  fired on one side only); **S=2 median per-parcel rel 0.62**,
  189/300 parcels over 0.5, worst CPU 8.72e7 vs GPU 4.40e11. The mitigation is
  that nothing downstream reads the magnitude (see the `s_psi` argument below),
  not that the disagreement is rare. `mu` is fine once measured as the unit
  direction it is: **per-column cosine ≥ 0.998 (S=1) / 0.99999997 (S=2)**; the
  element-wise `mu` max-rel of ~2.0 at S=1 is one sign-flipped near-zero
  component. Downstream, step 3 uses `epsil` only inside
  `s_psi = normalize(sigma·Σ nu + epsil·mu)`, where `epsil >> sigma` makes
  `s_psi ≈ mu` regardless of the magnitude. `compare_step2_backends.py` prints
  both the log10-ratio view of `epsil` and the per-column cosine of `mu`, and
  §7.3 requires them **reported** (not barred).
* Potential later win: per-subject member compaction by `s_lambda[s] != 0` for K3
  (tighter than `supp(theta)` at S > 1); warp-parallel `invad`.

## 9. Measured results (2026-09-03, RTX 5090 Laptop)

| | S=1 | S=2 |
|---|---:|---:|
| warm `Step2Pipeline.run()` | 0.32-0.36 s (was 6.9 s dense GPU / 71 s CPU) | 0.59-0.64 s (was 14.0 / 103 s) |
| `em_total` | 0.118 s, 42 `run_iter`, ~3.1 ms each incl. K7 (~2.5 ms steady) | 0.234 s, 42 calls |
| `load_inputs` / `session_ctor` / `init_device` / closure / save | 0.100 / 0.058 / 0.009 / 0.019 / 0.006 s | 0.174 / 0.086 / 0.017 / 0.063 / 0.006 s |
| per-launch: K3 / K6 / K8-10 / K4 per iter_m / K5 / K12 / K7 once | 0.34 / 0.35 / 0.40 / 0.17 / 0.15 / 0.22 / 25 ms | 0.85 / 0.44 / 0.40 / 0.31 / 0.19 / 0.22 / 23 ms |
| fresh process (cold interpreter, prewarm on a thread) | ~1.4 s | ~2.0 s |
| vs CPU: theta argmax flips / support-diff rows / dead-alive rows | 0 / 1 / 1303 = 1303 | 0 / 0 / 168 = 168 |
| vs CPU: `kappa` / first-inter `Record` rel / `sigma` max-rel | 0.0 / 4.0e-6 / 3.9e-3 | 0.0 / 1.4e-6 / 3.7e-4 |
| run-to-run | identical on every saved field | identical |
| step 3 labels from this prior vs from the CPU prior (sub-001, gpu_full) | identical (0 / 73 641) | — |
| the retired dense port, for comparison: flips / dead rows / step-3 label agreement | 4 744 / 5 528 / 93.5 % | 5 071 / 1 140 / — |

Kernel history of the perf pass (every step gated on bit-identical saved
Params): K3 dropped the shared-memory member staging (coalesced `__ldg` of the
row every thread walks anyway) and went to 256 threads × 1 byte with a 4-deep
member prefetch (0.80 → 0.33 ms); K6 uses 768-thread blocks (the 75-533-member
columns then take one pass) and an 8-byte prefetch (0.77 → 0.34 ms);
`widen_exact` is a warp per row (1.5 → 0.36 ms per 8192-row tile, K7 39 → 25
ms); `init_hard_labels` is register-tiled (22.9 → 8.5 ms); `connect_u` stages
the member list once per chunk (148 → 105 µs); `intra_flags_rel` is a block
with shared-memory staging (128 → 118 µs); the closure dots
`intra_sigma_dot` / `cost_terms_perl` are block trees (863/997 → 37/47 µs) while
`inter_eps_dot` stays serial (see §3 K13c). Full pytest suite: 429 passed,
10 skipped. Driver Mode B end-to-end (two subjects, gpu backends throughout):
19.7 s total, step 2 0.87 s, step 3 consumes the prior unchanged. These are
driver runs, so the default dense `theta` writer.

10-subject YS Mode B on this tree (2026-09-04, step 0/1 pinned `cpu` so every
arm's step-2 inputs are byte-identical, step 3 `gpu_full`): step 2 wall
`cpu` 538 s / the dense port 62 s / this backend 4.1 s. This backend vs `cpu`:
2 theta-argmax flips over 74 946 alive rows, 0 dead rows, `kappa` max-rel
5.4e-7, `mu` cos-min 0.999992, step-3 labels 99.05 % identical (7 156 of
749 460 cortex vertices). The dense port vs `cpu` on the same inputs: 4 783 flips,
29 dead rows, `kappa` max-rel 1.0e-2, 87.5 % step-3 label agreement —
identical on clean 8cb68a0, i.e. pre-existing. `Params_Final.mat` from the
candidate's `cpu` backend is byte-identical to clean HEAD's past the MAT header.
