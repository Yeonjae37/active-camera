"""Camera-selection policies on an error matrix E (frames x views) and the motion-constrained rail oracle."""

import math

import numpy as np

N_PHASE_BINS = 10


def move_profile(d, vmax, amax, dt):
    """Travelled fraction (0..1] at frames 1..n of a rest-to-rest trapezoidal move of d metres (None = no limit)."""
    v = math.inf if vmax is None else vmax
    a = math.inf if amax is None else amax
    if d == 0 or (math.isinf(v) and math.isinf(a)):
        return np.array([1.0])
    if math.isinf(a):
        T = d / v
    elif math.isinf(v) or d < v * v / a:  # triangular profile
        T = 2 * math.sqrt(d / a)
        v = a * T / 2
    else:
        T = d / v + v / a
    t = np.arange(1, max(1, math.ceil(T / dt - 1e-9)) + 1) * dt
    if math.isinf(a):
        x = v * t
    else:
        ta = v / a
        x = np.where(t < ta, 0.5 * a * t ** 2,
                     np.where(t < T - ta, 0.5 * a * ta ** 2 + v * (t - ta), d - 0.5 * a * np.clip(T - t, 0, None) ** 2))
    return np.minimum(x, d) / d


def constrained_oracle(E, pos, vmax, amax, dt):
    """Lowest-cost trajectory over a uniform position grid with rest-to-rest moves between any two positions;
    frames during a move are scored at the nearest grid position. Returns (mean cost, index per frame)."""
    T, P = E.shape
    step = pos[1] - pos[0]
    moves = {}
    for i in range(P):
        for j in range(P):
            if i != j:
                path = [i + int(np.sign(j - i)) * int(s)
                        for s in np.round(move_profile(abs(j - i) * step, vmax, amax, dt) * abs(j - i))]
                cost = np.zeros(T)  # cost of the move for every start frame t (frames t+1 .. t+n)
                for k, pk in enumerate(path[:T - 1]):
                    cost[:T - 1 - k] += E[1 + k:, pk]
                moves[i, j] = path, cost
    V = np.full((T, P), np.inf)
    V[0] = E[0]
    back = [[None] * P for _ in range(T)]
    end_cost, end_move = np.inf, None  # best trajectory that ends while still moving
    for t in range(T - 1):
        for i in range(P):
            if V[t, i] + E[t + 1, i] < V[t + 1, i]:
                V[t + 1, i], back[t + 1][i] = V[t, i] + E[t + 1, i], (t, i, [i])
            for j in range(P):
                if j == i:
                    continue
                path, cost = moves[i, j]
                c, n = V[t, i] + cost[t], len(path)
                if t + n <= T - 1:
                    if c < V[t + n, j]:
                        V[t + n, j], back[t + n][j] = c, (t, i, path)
                elif c < end_cost:
                    end_cost, end_move = c, (t, i, path[:T - 1 - t])
    traj = np.full(T, -1)
    t, i = T - 1, int(np.argmin(V[T - 1]))
    total = V[t, i]
    if end_cost < total:
        total, (t, i, path) = end_cost, end_move
        traj[t + 1:] = path
    traj[t] = i
    while t > 0:
        pt, pi, path = back[t][i]
        traj[pt + 1:pt + 1 + len(path)] = path
        t, i = pt, pi
        traj[t] = i
    return float(total / T), traj


def switch_stats(path, fps):
    path = np.asarray(path, float)
    sw = np.flatnonzero(np.diff(path) != 0)
    return {"switches": len(sw), "switches_per_s": len(sw) * fps / len(path),
            "median_run_frames": float(np.median(np.diff(np.r_[-1, sw, len(path) - 1]))),
            "total_travel": float(np.abs(np.diff(path)).sum())}


def phase_bins(phase):
    return np.minimum((np.nan_to_num(phase, nan=0) * N_PHASE_BINS).astype(int), N_PHASE_BINS - 1)


def bin_means(E, mask, bins):
    """Mean error per phase bin over the rows in mask (NaN for empty bins)."""
    return np.array([E[mask & (bins == b)].mean(0) if (mask & (bins == b)).any() else np.full(E.shape[1], np.nan)
                     for b in range(N_PHASE_BINS)])


def policies(E, views, ref, train, evaluate, phase, fps, numeric=True):
    """Policies on the rows `evaluate` (bool mask) of E. ref: index of the fixed reference view;
    train: rows used to pick the train-selected view and the phase-locked table."""
    Ee = E[evaluate]
    rows = np.arange(len(Ee))
    best = int(Ee.mean(0).argmin())
    picked = int(E[train].mean(0).argmin())
    bins = phase_bins(phase)
    table = bin_means(E, train, bins)
    predicted = table[bins[evaluate]]
    locked = None if np.isnan(predicted).any() else predicted.argmin(1)  # needs every phase bin in train
    oracle = Ee.argmin(1)
    stats = lambda idx: switch_stats(np.asarray(views, float)[idx], fps) if numeric else {}
    return {
        "reference_fixed": {"err": float(Ee[:, ref].mean()), "view": views[ref]},
        "best_fixed_posthoc": {"err": float(Ee[:, best].mean()), "view": views[best]},
        "best_fixed_train": {"err": float(Ee[:, picked].mean()), "view": views[picked]},
        "phase_locked_train": None if locked is None else {
            "err": float(Ee[rows, locked].mean()), **stats(locked), "views_per_phase_bin": [views[i] for i in table.argmin(1)]},
        "framewise_oracle": {"err": float(Ee.min(1).mean()), **stats(oracle)},
    }, predicted


def viewpoint_variance(E):
    """Share of error variance between views of the same frame; lag-1 autocorrelation of the view effect."""
    within = np.var(E, axis=1).mean()
    dm = E - E.mean(1, keepdims=True)
    ac = [np.corrcoef(dm[1:, j], dm[:-1, j])[0, 1] for j in range(E.shape[1])]
    return {"within_frame_variance_share": float(within / np.var(E)), "within_frame_sd_deg": float(np.sqrt(within)),
            "lag1_autocorr_of_view_effect": float(np.nanmean(ac))}


def phase_consistency(E, phase, cycle):
    """Correlation between gait cycles of the phase-bin x view error surface (bin means removed)."""
    bins = phase_bins(phase)
    surf = {}
    for c in np.unique(cycle[np.isfinite(phase)]):
        s = bin_means(E, cycle == c, bins)
        surf[int(c)] = (s - s.mean(1, keepdims=True)).ravel()
    ok = lambda a, b: np.isfinite(surf[a]) & np.isfinite(surf[b])
    return {f"{a}-{b}": float(np.corrcoef(surf[a][ok(a, b)], surf[b][ok(a, b)])[0, 1])
            for a in surf for b in surf if a < b and ok(a, b).sum() > 2}
