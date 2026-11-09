# =====================================================================
# Fast Cao E1/E2 (KD-tree, Chebyshev norm). Same quantities as the
# original cao_method translation; minutes instead of hours on ~40k points.
# Run in DK1_revision.ipynb after cells 1-2 (uses fit_raw / train_raw).
# =====================================================================
import numpy as np
from scipy.spatial import cKDTree

def _first_positive_nn(Y_pts, k0=32):
    """Nearest neighbour with distance > 0 (Chebyshev), as in the original:
    exact duplicates are skipped. Returns (index, distance) per point."""
    tree = cKDTree(Y_pts)
    n = len(Y_pts)
    idx = np.full(n, -1); dist = np.zeros(n)
    todo = np.arange(n); k = k0
    while len(todo) and k <= n:
        d, j = tree.query(Y_pts[todo], k=min(k, n), p=np.inf, workers=-1)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        pos = d > 0
        has = pos.any(axis=1)
        first = pos.argmax(axis=1)
        rows = np.where(has)[0]
        dmin = d[rows, first[rows]]
        # tie-break as in the original: lowest index among equally near neighbours
        tied = np.where(d[rows] == dmin[:, None], j[rows], np.iinfo(np.int64).max)
        idx[todo[rows]] = tied.min(axis=1)
        dist[todo[rows]] = dmin
        todo = todo[~has]; k *= 4
    return idx, dist          # idx = -1 only if every point is identical

def cao_fast(data, min_m=1, max_m=20, tau=1, verbose=True):
    data = np.asarray(data, dtype=float).ravel()
    N = len(data)
    E, Es = [], []
    for m in range(min_m, max_m + 1):
        n_pts = N - m * tau
        # rows = points, columns = delay coordinates (same as the original's Y[:, :n_points].T)
        Y = np.stack([data[i * tau: i * tau + n_pts] for i in range(m)], axis=1)
        j, d = _first_positive_nn(Y)
        ok = (j >= 0) & (d > 0)
        i_ok, j_ok = np.where(ok)[0], j[ok]
        nxt_i, nxt_j = data[i_ok + m * tau], data[j_ok + m * tau]
        d_next = np.maximum(d[ok], np.abs(nxt_i - nxt_j))          # Chebyshev distance in m+1 dims
        E.append(np.sum(d_next / d[ok]) / n_pts)
        Es.append(np.sum(np.abs(nxt_i - nxt_j)) / n_pts)
        if verbose:
            print(f"m={m}", flush=True)
    E, Es = np.array(E), np.array(Es)
    dims = np.arange(min_m, max_m)
    E1 = np.column_stack([dims, E[1:] / E[:-1]])
    E2 = np.column_stack([dims, Es[1:] / Es[:-1]])
    return E1, E2

def select_m(E1, e=0.1, min_m=1):
    """Same automated rule as the original MATLAB-translated block."""
    v = E1[:, 1]; delt = np.abs(np.diff(v)); num = np.where(delt < e)[0]
    if not len(num):
        return 0
    tail = delt[num[0]:]; num1 = num[0] + int(np.argmax(tail)); e = np.mean(delt[num1:])
    for kk in range(num1, len(v)):
        if kk + 2 < len(v):
            d1, d2, d3 = abs(v[kk] - v[kk - 1]), abs(v[kk + 1] - v[kk]), abs(v[kk + 2] - v[kk + 1])
            if d1 > d2 > d3 and d2 < e:
                return kk + min_m
    return 0

# ---------------- DK1 check: Cao on the TRAINING price differences ----------------
E1, E2 = cao_fast(train_raw, min_m=1, max_m=20, tau=1)
m_sel = select_m(E1)
print("\n  m    E1(m)    E2(m)")
for (m, e1), (_, e2) in zip(E1, E2):
    print(f"{int(m):3d}  {e1:7.4f}  {e2:7.4f}")
print(f"\nSelected embedding dimension (automated rule): m = {m_sel}")
print("E2 close to 1 for all m  => predominantly stochastic dynamics" if np.all(np.abs(E2[:, 1] - 1) < 0.1)
      else "E2 departs from 1 for some m => some deterministic structure")
