"""Side-by-side photos of a sweep: reference / best / worst view per frame, with the predicted (NLF) and GT legs.

Uses the analysis outputs of analyze.py (per_image_<mode>.csv, nlf_joints3d_<mode>.npz); no inference.

    python -m src.simulation.figures output/03_orbit_azimuth/capture [--frames 91 136] [--n 3]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import VIEW
from .common import ROOT, draw_legs, load_sweep, person_crop, project, rgb

GREEN, RED, BLACK = "#1a7f37", "#c62828", "#222222"


def photo_grid(rows, labels, title, path, size=3.6):
    """rows: lists of (image, caption, colour); coloured frames mark best (green) / worst (red)."""
    fig, axes = plt.subplots(len(rows), max(map(len, rows)), figsize=(size * max(map(len, rows)), size * 1.22 * len(rows)),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(row):
                ax.axis("off")
                continue
            image, caption, colour = row[c]
            ax.imshow(image)
            ax.set_title(caption, fontsize=10, color=colour, fontweight="normal" if colour == BLACK else "bold")
            for sp in ax.spines.values():
                sp.set(edgecolor=colour, linewidth=1 if colour == BLACK else 3)
        axes[r, 0].set_ylabel(labels[r], fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--analysis", help="default <sweep>/../analysis")
    ap.add_argument("--mode", default="raw")
    ap.add_argument("--frames", nargs="*", type=int, help="default: the --n frames with the largest best-worst spread")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    meta, caps = load_sweep(args.sweep)
    kind = meta.get("experiment", "pan_tilt")
    key, ref = VIEW[kind]
    res = ROOT / args.analysis if args.analysis else caps[0]["path"].parent.parent / "analysis"
    df = pd.read_csv(res / f"per_image_{args.mode}.csv")
    df["view"] = [(c["pan"], c["tilt"]) for c in caps]  # same order as the captures
    z = np.load(res / f"nlf_joints3d_{args.mode}.npz")
    j3 = dict(zip(z["files"], z["joints3d"].astype(float)))
    by_file = {c["file"]: c for c in caps}
    spread = df.groupby("frame").mean_flex_err_deg.agg(lambda e: e.max() - e.min())
    frames = args.frames or sorted(spread.nlargest(args.n).index)

    def photo(r, head, colour):
        c = by_file[r.file]
        uv, _ = project(c, c["gt"])
        caption = f"{head}\nL {r.left_flex_err_deg:.1f}  R {r.right_flex_err_deg:.1f}  (mean {r.mean_flex_err_deg:.1f} deg)"
        return person_crop(draw_legs(rgb(c["path"]), j3[r.file], uv, c["K"]), uv), caption, colour

    rows, labels = [], []
    for f in frames:
        g = df[df.frame == f]
        best, worst = g.loc[g.mean_flex_err_deg.idxmin()], g.loc[g.mean_flex_err_deg.idxmax()]
        reference = g[g[key].apply(lambda v: v == ref)].iloc[0]
        rows.append([photo(reference, f"reference: {reference[key]}", BLACK),
                     photo(best, f"best: {best[key]}", GREEN), photo(worst, f"worst: {worst[key]}", RED)])
        labels.append(f"frame {f}\nGT L {g.left_flex_gt_deg.iloc[0]:.0f} / R {g.right_flex_gt_deg.iloc[0]:.0f} deg")
    path = res / f"best_worst_{args.mode}.png"
    photo_grid(rows, labels, f"{kind} ({key}): reference vs best vs worst view, |3D knee flexion error|", path)
    print("->", path)


if __name__ == "__main__":
    main()
