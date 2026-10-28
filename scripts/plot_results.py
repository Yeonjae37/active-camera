"""Plot a Pose2Sim project's results: joint angles (.mot) and raw vs filtered markers (.trc).

Usage (from repo root):
    uv run python scripts/plot_results.py                       # output/pose2sim_demo
    uv run python scripts/plot_results.py output/xxx --markers RKnee LAnkle
PNGs go to <project>/plots/.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from biomechanics import read_mot, read_trc
from evaluation.plots import GAIT_COORDS, plot_angles, plot_markers

REPO_ROOT = Path(__file__).resolve().parents[1]


def trc_label(path: Path) -> str:
    # pose2sim_demo_1-97_filt_butterworth.trc -> filt_butterworth
    s = path.stem
    return s[s.index("_filt") + 1:] if "_filt" in s else "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path, nargs="?", default=REPO_ROOT / "output" / "pose2sim_demo")
    ap.add_argument("--coords", nargs="+", default=GAIT_COORDS)
    ap.add_argument("--markers", nargs="+", default=["RHip", "RKnee", "RAnkle"])
    args = ap.parse_args()

    out = args.project / "plots"
    out.mkdir(exist_ok=True)

    for mot in sorted(args.project.glob("kinematics/*.mot")):
        fig = plot_angles({mot.stem: read_mot(mot)}, args.coords, title=mot.name)
        fig.savefig(out / f"angles_{mot.stem}.png", dpi=120)

    # raw vs filtered (LSTM-augmented files have different markers, so they are skipped here)
    trcs = {trc_label(p): read_trc(p) for p in sorted(args.project.glob("pose-3d/*.trc")) if "LSTM" not in p.stem}
    if trcs:
        fig = plot_markers(trcs, args.markers, title="raw vs filtered")
        fig.savefig(out / "markers_raw_vs_filt.png", dpi=120)

    for p in sorted(out.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
