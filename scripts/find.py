
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AMASS_DIR = ROOT / "data/amass/CMU/CMU"

results = []

for path in AMASS_DIR.rglob("*_poses.npz"):
    try:
        with np.load(path, allow_pickle=False) as data:
            trans = data["trans"]
            fps = float(data["mocap_framerate"])

        if fps <= 0 or len(trans) < 2:
            continue

        duration = (len(trans) - 1) / fps

        xy = trans[:, :2]

        displacement = np.linalg.norm(
            xy[-1] - xy[0]
        )

        path_length = np.linalg.norm(
            np.diff(xy, axis=0),
            axis=1
        ).sum()

        straightness = (
            displacement / path_length
            if path_length > 0
            else 0.0
        )

        if duration >= 5.0 and displacement >= 3.0:
            results.append((
                displacement,
                duration,
                path_length,
                straightness,
                path.relative_to(AMASS_DIR)
            ))

    except (OSError, KeyError, ValueError):
        continue

results.sort(reverse=True)

print("Long motion candidates:\n")

for displacement, duration, length, straightness, path in results[:30]:
    print(
        f"{str(path):35s} "
        f"duration={duration:5.1f}s  "
        f"displacement={displacement:5.2f}m  "
        f"path={length:5.2f}m  "
        f"straightness={straightness:.2f}"
    )

print(f"\nTotal candidates: {len(results)}")
