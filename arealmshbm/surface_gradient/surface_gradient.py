"""surface_gradient.py

Per-hemisphere, per-column tangent-plane LS gradient magnitude.

Numba @njit port of HCP Workbench
``src/Algorithms/AlgorithmMetricGradient.cxx`` (multi-column branch,
regression method with the "unrolled" geodesic-magnitude correction
and per-vertex-area weighting). The public symbol ``cifti_gradient``
is itself the @njit Dispatcher — there is no Python wrapper outside
the JIT'd entry point. fp32 throughout the memory-traffic path; the
3x3 normal-equations solve is done in fp64 for stability via an
explicit Cramer's rule (no np.linalg.solve per vertex).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Width of the 1-ring neighbor table (``nbors`` is (N_v, _MAX_DEG), -1
# padded) and of the per-vertex stack buffers in the prange body.
# fsaverage-class triangulations top out at degree 6-8.
_MAX_DEG = 32


# ─────────────────────────────────────────────────────────────────────────
# Tangent-basis helper
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, inline='always')
def _seed_xhat_yhat(nx, ny, nz):
    """Workbench convention for picking a tangent basis given a normal.

    Returns (xhat_x, xhat_y, xhat_z, yhat_x, yhat_y, yhat_z) as fp32.
    """
    if abs(nx) > abs(ny):
        sx = 0.0; sy = 1.0; sz = 0.0
    else:
        sx = 1.0; sy = 0.0; sz = 0.0
    # xhat = normal x seed
    xx = ny * sz - nz * sy
    xy = nz * sx - nx * sz
    xz = nx * sy - ny * sx
    xn = np.sqrt(xx * xx + xy * xy + xz * xz)
    if xn > 0.0:
        inv = 1.0 / xn
        xx *= inv; xy *= inv; xz *= inv
    # yhat = normal x xhat
    yx = ny * xz - nz * xy
    yy = nz * xx - nx * xz
    yz = nx * xy - ny * xx
    yn = np.sqrt(yx * yx + yy * yy + yz * yz)
    if yn > 0.0:
        inv = 1.0 / yn
        yx *= inv; yy *= inv; yz *= inv
    return (np.float32(xx), np.float32(xy), np.float32(xz),
            np.float32(yx), np.float32(yy), np.float32(yz))


# ─────────────────────────────────────────────────────────────────────────
# Per-hemisphere gradient magnitude — parallel over vertices
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def _hemi_gradient(deg, nbors, vn, va, verts, data, roi, out):
    """Fill ``out[v, k]`` with gradient magnitude. Medial rows stay zero.

    The four mesh tables ship in the step-0 inputs bundle
    (``Step0Inputs.{lh,rh}_grad_prep``):
        deg   : (N_v,)            int32   — 1-ring degree
        nbors : (N_v, _MAX_DEG)   int32   — 1-ring neighbor indices, -1 padded
        vn    : (N_v, 3)          fp32    — unit vertex normals
        va    : (N_v,)            fp32    — 1/3 of incident-face area sum
    """
    n_verts = verts.shape[0]
    K = data.shape[1]

    for i in prange(n_verts):
        if not roi[i]:
            continue
        d_full = deg[i]
        if d_full < 2:
            continue
        nx = vn[i, 0]; ny = vn[i, 1]; nz = vn[i, 2]
        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            continue

        # ROI-filter neighbors into stack-local arrays.
        nb = np.empty(_MAX_DEG, dtype=np.int32)
        m = 0
        for kk in range(d_full):
            j = nbors[i, kk]
            if roi[j]:
                nb[m] = j
                m += 1
        if m < 2:
            continue

        xhx, xhy, xhz, yhx, yhy, yhz = _seed_xhat_yhat(nx, ny, nz)
        cvx = verts[i, 0]; cvy = verts[i, 1]; cvz = verts[i, 2]

        # Per-neighbor scratch (xmag, ymag, weight, fallback support).
        xmag = np.empty(_MAX_DEG, dtype=np.float32)
        ymag = np.empty(_MAX_DEG, dtype=np.float32)
        ww = np.empty(_MAX_DEG, dtype=np.float32)
        # Magnitudes consulted only on the rare singular fallback below.
        unrollM = np.empty(_MAX_DEG, dtype=np.float32)
        mag2dM = np.empty(_MAX_DEG, dtype=np.float32)

        a00 = 0.0; a01 = 0.0; a02 = 0.0
        a11 = 0.0; a12 = 0.0; a22 = 0.0
        # 3x3 normal-equations accumulator (fp64 for stability).
        for q in range(m):
            j = nb[q]
            ox = verts[j, 0] - cvx
            oy = verts[j, 1] - cvy
            oz = verts[j, 2] - cvz
            origMag = np.sqrt(ox * ox + oy * oy + oz * oz)
            opp = ox * nx + oy * ny + oz * nz
            unrollMag = origMag
            if origMag > 0.0 and abs(opp) > 0.035 * origMag:
                r = opp / origMag
                if r > 1.0:
                    r = 1.0
                elif r < -1.0:
                    r = -1.0
                unrollMag = origMag * np.arcsin(r) * origMag / opp
            xm0 = ox * xhx + oy * xhy + oz * xhz
            ym0 = ox * yhx + oy * yhy + oz * yhz
            mag2d = np.sqrt(xm0 * xm0 + ym0 * ym0)
            if mag2d > 0.0:
                s = unrollMag / mag2d
                xm = xm0 * s
                ym = ym0 * s
            else:
                xm = 0.0
                ym = 0.0
            wj = va[j]
            xmag[q] = np.float32(xm)
            ymag[q] = np.float32(ym)
            ww[q] = wj
            unrollM[q] = np.float32(unrollMag)
            mag2dM[q] = np.float32(mag2d)
            wj64 = float(wj)
            xm64 = float(xm)
            ym64 = float(ym)
            a00 += xm64 * xm64 * wj64
            a01 += xm64 * ym64 * wj64
            a02 += xm64 * wj64
            a11 += ym64 * ym64 * wj64
            a12 += ym64 * wj64
            a22 += wj64
        a22 += float(va[i])

        # Cramer's rule on a symmetric 3x3 system.
        # |A| = a00(a11 a22 - a12^2) - a01(a01 a22 - a12 a02)
        #       + a02(a01 a12 - a11 a02)
        det = (a00 * (a11 * a22 - a12 * a12)
               - a01 * (a01 * a22 - a12 * a02)
               + a02 * (a01 * a12 - a11 * a02))

        # Per-K solve
        for k in range(K):
            f_v = data[i, k]
            b0 = 0.0; b1 = 0.0; b2 = 0.0
            for q in range(m):
                j = nb[q]
                tempf = float(data[j, k] - f_v)
                wt = float(ww[q]) * tempf
                b0 += float(xmag[q]) * wt
                b1 += float(ymag[q]) * wt
                b2 += float(ww[q]) * tempf
            sol_a = 0.0
            sol_b = 0.0
            ok = False
            if det != 0.0:
                inv_det = 1.0 / det
                # A inverse * B (symmetric A)
                # cofactors for the first two rows of solution suffice
                # sol_a = det( [[b0,a01,a02],[b1,a11,a12],[b2,a12,a22]] ) / det
                # sol_b = det( [[a00,b0,a02],[a01,b1,a12],[a02,b2,a22]] ) / det
                num_a = (b0 * (a11 * a22 - a12 * a12)
                         - a01 * (b1 * a22 - a12 * b2)
                         + a02 * (b1 * a12 - a11 * b2))
                num_b = (a00 * (b1 * a22 - a12 * b2)
                         - b0 * (a01 * a22 - a12 * a02)
                         + a02 * (a01 * b2 - b1 * a02))
                sol_a = num_a * inv_det
                sol_b = num_b * inv_det
                if (np.isfinite(sol_a) and np.isfinite(sol_b)):
                    ok = True
            if not ok:
                # Workbench fallback: weighted finite-difference average.
                # tempf2 = tempf / (unrollMag * mag2d);
                # sol_a = sum(xmag_scaled * tempf2 * w) / sum(w).
                tot = 0.0
                acc_a = 0.0
                acc_b = 0.0
                for q in range(m):
                    j = nb[q]
                    denom = float(unrollM[q]) * float(mag2dM[q])
                    wj = float(ww[q])
                    tempf = float(data[j, k] - f_v)
                    if denom > 0.0:
                        t2 = tempf / denom
                        acc_a += float(xmag[q]) * t2 * wj
                        acc_b += float(ymag[q]) * t2 * wj
                    tot += wj
                if tot > 0.0:
                    sol_a = acc_a / tot
                    sol_b = acc_b / tot
                else:
                    sol_a = 0.0
                    sol_b = 0.0

            gx = float(xhx) * sol_a + float(yhx) * sol_b
            gy = float(xhy) * sol_a + float(yhy) * sol_b
            gz = float(xhz) * sol_a + float(yhz) * sol_b
            out[i, k] = np.float32(np.sqrt(gx * gx + gy * gy + gz * gz))


# ─────────────────────────────────────────────────────────────────────────
# Public entry point — itself the @njit dispatcher
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def cifti_gradient(data,
                   lh_verts, lh_deg, lh_nbors, lh_vn, lh_va,
                   rh_verts, rh_deg, rh_nbors, rh_vn, rh_va,
                   medial_mask):
    """Drop-in replacement for ``wb_command -cifti-gradient`` on the
    surface portion of a CIFTI dtseries (no subcortical voxels).

    Mesh-only inputs (``{lh,rh}_{deg,nbors,vn,va}``) are the
    ``Step0Inputs.{lh,rh}_grad_prep`` tables; the per-call hot path
    stays purely data-bound.

    Returns
    -------
    grads : (N_cortex, K) fp32 ndarray
    """
    n_lh = lh_verts.shape[0]
    n_rh = rh_verts.shape[0]
    n_full = n_lh + n_rh
    K = data.shape[1]

    # Cortex bool mask (medial_mask is uint8/bool: 1 = medial).
    cortex = np.empty(n_full, dtype=np.bool_)
    n_cortex = 0
    for i in range(n_full):
        c = not medial_mask[i]
        cortex[i] = c
        if c:
            n_cortex += 1

    # Scatter cortex-only data to full mesh.
    full = np.zeros((n_full, K), dtype=np.float32)
    pos = 0
    for i in range(n_full):
        if cortex[i]:
            for k in range(K):
                full[i, k] = data[pos, k]
            pos += 1

    lh_full = full[:n_lh]
    rh_full = full[n_lh:]
    lh_roi = cortex[:n_lh]
    rh_roi = cortex[n_lh:]

    lh_grads = np.zeros((n_lh, K), dtype=np.float32)
    rh_grads = np.zeros((n_rh, K), dtype=np.float32)
    _hemi_gradient(lh_deg, lh_nbors, lh_vn, lh_va, lh_verts, lh_full, lh_roi, lh_grads)
    _hemi_gradient(rh_deg, rh_nbors, rh_vn, rh_va, rh_verts, rh_full, rh_roi, rh_grads)

    # Re-mask: gather cortex rows from the concatenated full output.
    out = np.empty((n_cortex, K), dtype=np.float32)
    pos = 0
    for i in range(n_lh):
        if lh_roi[i]:
            for k in range(K):
                out[pos, k] = lh_grads[i, k]
            pos += 1
    for i in range(n_rh):
        if rh_roi[i]:
            for k in range(K):
                out[pos, k] = rh_grads[i, k]
            pos += 1
    return out
