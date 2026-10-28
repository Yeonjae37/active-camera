"""Isaac Sim camera poses -> Pose2Sim Calib.toml (OpenCV convention). Pure numpy, no Isaac Sim needed.

Input per camera: camera_poses.npy-style 4x4 (USD row-vector camera-to-world, camera looks along -Z, +Y up)
and a metadata.json-style intrinsics dict. Output world frame = Isaac world (Z-up, meters); Pose2Sim
converts to Y-up itself when it writes .trc.
"""

from pathlib import Path

import numpy as np

USD_TO_CV = np.diag([1.0, -1.0, -1.0])  # camera axes: USD (x right, y up, -z fwd) -> OpenCV (x right, y down, z fwd)


def usd_to_opencv_extrinsics(c2w_row: np.ndarray):
    """Returns (R, t) with x_cam = R @ x_world + t (OpenCV world-to-camera)."""
    T = np.asarray(c2w_row, dtype=np.float64).T  # column-vector camera-to-world
    R_c2w = T[:3, :3] @ USD_TO_CV
    R = R_c2w.T
    return R, -R @ T[:3, 3]


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> Rodrigues vector (what cv2.Rodrigues returns)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3)
    if np.pi - angle < 1e-6:  # near 180 deg: R = 2 n n^T - I
        i = int(np.argmax(np.diag(R)))
        axis = R[i] / 2.0
        axis[i] = np.sqrt((R[i, i] + 1.0) / 2.0)
        axis[[j for j in range(3) if j != i]] /= axis[i]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(angle)) * angle


def project(points_w: np.ndarray, c2w_row: np.ndarray, intr: dict) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixels, for sanity checks against the rendered images."""
    R, t = usd_to_opencv_extrinsics(c2w_row)
    pc = points_w @ R.T + t
    return np.stack([intr["fx"] * pc[:, 0] / pc[:, 2] + intr["cx"],
                     intr["fy"] * pc[:, 1] / pc[:, 2] + intr["cy"]], axis=1)


def _fmt(v):
    return "[ " + ", ".join(repr(float(x)) for x in v) + "]"


def write_calib_toml(path, cameras: dict):
    """cameras: {name: (c2w_row 4x4, intrinsics dict)}, written in sorted name order (Pose2Sim pairs
    calibration entries with the sorted video folders, so name them cam01, cam02, ...)."""
    lines = []
    for name in sorted(cameras):
        c2w, intr = cameras[name]
        R, t = usd_to_opencv_extrinsics(c2w)
        K = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
        lines += [f"[{name}]", f'name = "{name}"', f"size = {_fmt([intr['width'], intr['height']])}",
                  "matrix = [ " + ", ".join(_fmt(r) for r in K) + "]",
                  "distortions = [ 0.0, 0.0, 0.0, 0.0]",
                  f"rotation = {_fmt(rotvec(R))}", f"translation = {_fmt(t)}", "fisheye = false", ""]
    lines += ["[metadata]", "adjusted = false", "error = 0.0", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def calib_from_recordings(out, rec_dirs, frame=-1):
    """One recording folder per static camera (camera_poses.npy, metadata.json, PNGs) -> Calib.toml.
    Intrinsics are rescaled to the size of the saved PNGs (the viewport captures were 1280x720 while
    metadata.json says 1920x1080)."""
    import json
    cameras = {}
    for i, d in enumerate(map(Path, rec_dirs), 1):
        poses = np.load(d / "camera_poses.npy")
        intr = dict(json.loads((d / "metadata.json").read_text())["intrinsics"])
        w, h = _png_size(sorted(d.glob("*.png"))[0])
        sx, sy = w / intr["width"], h / intr["height"]
        intr.update(width=w, height=h, fx=intr["fx"] * sx, fy=intr["fy"] * sy, cx=intr["cx"] * sx, cy=intr["cy"] * sy)
        drift = np.abs(poses[:, 3, :3] - poses[frame, 3, :3]).max()
        if drift > 1e-3:
            print(f"warning: {d} camera moved {drift * 1000:.1f} mm during the take; using frame {frame}")
        cameras[f"cam{i:02d}"] = (poses[frame], intr)
    write_calib_toml(out, cameras)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=calib_from_recordings.__doc__)
    ap.add_argument("out", help="e.g. output/step3/calibration/Calib.toml")
    ap.add_argument("recordings", nargs="+", help="one recording folder per camera, in cam01, cam02, ... order")
    ap.add_argument("--frame", type=int, default=-1, help="pose to use (default: last, after the drives settled)")
    a = ap.parse_args()
    calib_from_recordings(a.out, a.recordings, a.frame)
