"""Overlay plots for kinematics results, one color per condition label.

Every function takes {label: data} so the same call compares filtered vs raw,
estimate vs GT, or fixed vs active camera. Load data with biomechanics.io first.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biomechanics.io import Trc

GAIT_COORDS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _grid(n: int, ncols: int, w: float = 4.0, h: float = 2.6):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False, sharex=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _finish(fig, axes, labels: Sequence[str], title: str | None):
    handles = [plt.Line2D([], [], color=_color(i), label=l) for i, l in enumerate(labels)]
    fig.legend(handles=handles, loc="upper right", ncols=len(labels), frameon=False)
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_angles(mots: dict[str, pd.DataFrame], coords: Sequence[str] = GAIT_COORDS,
                ncols: int = 2, title: str | None = None):
    """Joint-angle waveforms (deg vs s) from read_mot DataFrames."""
    fig, axes = _grid(len(coords), ncols)
    for ax, c in zip(axes, coords):
        for i, (label, df) in enumerate(mots.items()):
            if c in df:
                ax.plot(df.index, df[c], color=_color(i), lw=1.2)
        ax.set_title(c, fontsize=9)
        ax.set_ylabel("deg")
    for ax in axes[-ncols:]:
        ax.set_xlabel("time (s)")
    return _finish(fig, axes, list(mots), title)


def plot_markers(trcs: dict[str, Trc], markers: Sequence[str], title: str | None = None):
    """X/Y/Z trajectories, one row per marker, from read_trc results."""
    fig, axes = _grid(len(markers) * 3, 3)
    for r, m in enumerate(markers):
        for k, axis in enumerate("XYZ"):
            ax = axes[r * 3 + k]
            for i, trc in enumerate(trcs.values()):
                if m in trc.markers:
                    ax.plot(trc.time, trc.marker(m)[:, k], color=_color(i), lw=1.2)
            ax.set_title(f"{m} {axis}", fontsize=9)
            if k == 0:
                ax.set_ylabel(next(iter(trcs.values())).units)
    for ax in axes[-3:]:
        ax.set_xlabel("time (s)")
    return _finish(fig, axes, list(trcs), title)


def angle_errors(est: pd.DataFrame, ref: pd.DataFrame, coords: Sequence[str] = GAIT_COORDS) -> pd.DataFrame:
    """RMSE / mean bias (deg) per coordinate after aligning est onto ref's time stamps."""
    rows = {}
    for c in coords:
        if c in est and c in ref:
            e = np.interp(ref.index, est.index, est[c]) - ref[c].to_numpy()
            rows[c] = {"rmse": float(np.sqrt(np.nanmean(e**2))), "bias": float(np.nanmean(e))}
    return pd.DataFrame(rows).T
