# Per-subject RSFC profile on-disk format

`profiles_raw/sub<S>/sub<S>_<targ>_roi<seed>.profile.b2nd` is the
production format for the binary 0/1 RSFC profile that step1 emits and
step1's avg_profiles / step2's BOLD loader / step3's `fetch_data`
consume. One on-disk format: **bitpacked uint8**.

## Format

* On-disk shape `(T, N, ⌈D/8⌉)` uint8 — 1 bit/cell, LSB-first packing
  along D (cell `d` ↔ bit `(d & 7)` of byte `(d >> 3)`), matching
  `numpy.packbits(bitorder='little')`.
* b2nd vlmeta carries `format_version='bitpacked_uint8_v1'` and the
  unpacked length `D_unpacked` (so readers can recover the exact `D`
  given only `⌈D/8⌉` bytes per row), plus `bit_order='little'` for
  documentation.
* Padding bits past `D` in the last byte are zero (handled by
  `numpy.packbits` and guaranteed by the writer).
* Codec: LZ4 + bitshuffle, clevel 5. Bitshuffle on packed bytes is
  roughly neutral but LZ4 still picks up the long runs at medial wall
  / sparse-active regions that survive packing.

## Writing: one shot or one session at a time

Two writers, one format. `write_subject_profile_tnd(path, arr_tnd)`
takes the whole unpacked binary `(T, N, D)` subject and packs it
(the CPU stage pipeline's writer); `SubjectProfileStreamWriter(path,
T=, N=, D_unpacked=)` takes already-packed `(N, ⌈D/8⌉)` slabs a
**session** at a time — `write_session(t, slab)` compresses exactly
chunk `t`, `close()` finalises (the GPU leaf's writer, and the way
to write packed bytes verbatim). Because the chunk shape has always been
one session, the two produce the same payload: same `chunks`, same
auto-computed `blocks`, same LZ4-5 + bitshuffle `cparams`, same three
vlmeta keys. Readers cannot tell them apart, and there is no
`format_version` bump — nothing the tag pins has changed.

`format_version` is stamped **last**, by `close()`, after every session
chunk has been written (`D_unpacked` / `bit_order` are stamped up
front). Since `format_version` is exactly what `_open_and_verify` gates
on, a frame whose `close()` never ran — the process died mid-subject —
is *rejected* by the readers rather than served with zero-filled
sessions. `abort()` (which the context manager calls on an exception)
removes the partial file outright.

Chunk `t` is written by index, not appended, so sessions may be
finished out of order; `close()` refuses, and removes the file, if any
session was never written.

The streaming writer exists so step 1's fused GPU path can hide the
compression: each session's packed slab is D2H'd into its final place
in the pinned whole-subject block the moment its pack kernel is issued
and handed (with a CUDA event) to a single writer thread, so the
compression of session `t` overlaps the compute of `t + 1` and the join
at the end owes one chunk instead of the whole subject.

## Reading

| API                                          | Returns                          | Notes |
|----------------------------------------------|----------------------------------|-------|
| `open_subject_profile_packed_tnd(p)`         | `blosc2.NDArray` (lazy, packed)  | `arr[t] -> (N, ⌈D/8⌉) uint8`, zero unpack cost. Carries `.D_unpacked` (int). |
| `read_subject_profile_packed_tnd(p)`         | `(packed, D_unpacked)`           | One-shot full-load packed. |

Only bit-packed readers are exposed — the legacy fp32-view path
(`read_subject_profile_tnd` / `open_subject_profile_tnd` /
`SubjectProfileFp32View`) was removed 2026-06. Every consumer fuses
`bit-unpack → demean → L2-norm` in a single numba/RawKernel pass
straight from the packed bytes; the unpack-to-fp32-then-normalize
reference would otherwise materialize a ~2.3 GB (T, N, D) intermediate
at fsa6 T=6. (The step2 CPU stream-mode loader still allocates a
small ~72 MB (N, T, ⌈D/8⌉) **packed** transient per subject visit
so the fused kernel can consume the whole subject's packed bytes in
one pass — that's the buffer the kernel reads, not a widened fp32
copy, so it stays at the 1-bit/cell density.)

## Consumers

The .b2nd format is the **sole** on-disk BOLD source — every consumer
discovers it via `cohort.json` and reads it directly. No legacy
nii.gz fallback.

Step 2's `gpu` backend reads the packed bytes straight into its device
cache (`step2_io/sparse_inputs.py`, `bold_reader`); the `cpu` backend's
`SubjectProfileLoader` widens them through the fused numba kernel.

Step3's `fetch_data` always returns `(N, T, ⌈D/8⌉) uint8` plus
`D_unpacked`; each backend's :class:`VmfClusteringSession` runs its
own unpack+normalize on this payload:

* `gpu_full`: on-device fused `bit-unpack → demean → L2-norm` kernel
  `_normalize_bold_NTD_from_packed` in
  [`vmf_clustering_gpu.py`](arealmshbm/vmf_clustering/vmf_clustering_gpu.py).
  H2D ships packed bytes only — 8× smaller than the fp32 alternative.
* `cpu` / `gpu_elambda`: host numba kernel
  `_normalize_session_bitpacked_numba` in
  [`bitpacked_norm.py`](arealmshbm/data_io/bitpacked_norm.py), called
  per session from
  [`unpack_normalize_packed_NTD_host`](arealmshbm/data_io/bitpacked_norm.py)
  in the CPU Session's `__init__`. Output is bit-identical to the
  naive "unpack to fp32 + standard per-row demean + L2-norm" reference
  on binary input (proof: popcount in fp64 == Σ fp32(0/1); identical
  fp64 sumsq accumulation order).

Step1's `avg_profiles` subgraph reads packed bytes and accumulates over
set bits without ever materializing a fp32 BOLD slab:

* CPU: numba kernel `_accum_packed_session_inplace_kernel` (in
  [`avg_profiles/_kernels.py`](arealmshbm/avg_profiles/_kernels.py))
  reads `(V_h, ⌈D/8⌉) uint8` per (session, hemi) slab and adds
  `1.0f` to the running fp32 accumulator for each set bit.
* GPU: CUDA RawKernel `accum_packed_session_NhDb` (lazy-compiled in
  [`avg_profiles_gpu.py`](arealmshbm/avg_profiles/avg_profiles_gpu.py))
  — one block per hemi vertex, threads stride over D_bytes, no
  atomics (each (n, d) slot is owned by exactly one thread).

Output is bit-identical to a naive "fp32-unpack-then-add" reference.
Proof: every summand is 0.0f or 1.0f exactly, partial sums never
exceed S·T ≤ 1200 (well under 2^24), so fp32 adds are integer-exact
and order-independent.

## End-to-end correctness

The bit-packed representation is information-preserving for binary
input: the CPU backend's fused widen kernel reproduces an fp32
unpack-then-normalise bit for bit (proof above), so `Params_Final.mat`
does not depend on which representation reached the kernel; the step-2
`gpu` backend consumes the packed bytes directly, and its numerics
contract against the CPU reference is
[`step2_sparse_design.md`](step2_sparse_design.md) §7/§9 (not bit-exact
across backends).

## Disk size + wall

Reference cohort (S=3 fsa6 T=2 D=1175):

| metric                                     | bitpacked .b2nd |
|--------------------------------------------|----------------:|
| total .b2nd size (3 subjects)              | 48.4 MB         |
| step2 CPU `total`                          | ~20 s           |

Step 2's `gpu` backend walls (0.32-0.36 s at S=1, 16 s for the 40-subject
cohort) are in [`step2_flow_and_subgraphs.md`](step2_flow_and_subgraphs.md)
§ Wall. Step 3
single-subject (sub-001 / fsa6 / T=6 / L=300) on the `gpu_full` backend
runs in ~4.13 s total / ~0.12 s fetch_data / ~0.16 s session_init.
