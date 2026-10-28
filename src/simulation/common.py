"""Shared data access and geometry for the rendered sweeps (isaac/capture_sweep.py output)."""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
GT_NPZ = ROOT / "output/walking_data.npz"
LEGS = {"left": (1, 4, 7), "right": (2, 5, 8)}  # SMPL-24 hip, knee, ankle
TORSO = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]  # rigid alignment for the leg-identity diagnostic
GEOMETRY = ["rail_position_m", "orbit_azimuth_deg", "view_azimuth_deg", "view_azimuth_walk_deg",
            "view_elevation_deg", "camera_to_pelvis_m", "camera_to_pelvis_horizontal_m", "camera_tilt_deg"]


def load_sweep(sweep_dir):
    """metadata.json -> (meta, captures sorted by frame then view). Missing geometry keys -> None."""
    sweep_dir = Path(sweep_dir)
    meta = json.loads((sweep_dir / "metadata.json").read_text())
    caps = []
    for c in meta["captures"]:
        gap = c.get("joint_depth_gap_m") or [None] * 24
        caps.append({
            "file": c["file"], "path": sweep_dir / c["file"], "frame": int(c["frame"]),
            "pan": int(round(c["pan_offset_deg"])), "tilt": int(round(c["tilt_offset_deg"])),
            "c2w": np.asarray(c["camera_to_world"], float),  # USD row-vector, camera looks down -Z
            "K": np.asarray(c["intrinsics_K"], float), "gt": np.asarray(c["gt_joints_world"], float),
            "gap": np.array([np.nan if v is None else v for v in gap], float),
            **{k: c.get(k) for k in GEOMETRY},
        })
    view = {"rail": "rail_position_m", "orbit": "orbit_azimuth_deg"}.get(meta.get("experiment"))
    caps.sort(key=lambda c: (c["frame"], c[view] if view else (c["pan"], c["tilt"])))
    return meta, caps


def project(cap, points):
    """World points -> (uv px, depth). USD row-vector camera-to-world, -Z forward -> OpenCV pixels."""
    p = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))] @ np.linalg.inv(cap["c2w"])
    x, y, z = p[:, 0], -p[:, 1], -p[:, 2]
    K = cap["K"]
    return np.c_[K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]], z


def knee_flexion(joints, side):
    """180 - angle(hip-knee, ankle-knee) in degrees; 0 = straight leg. Rotation/scale invariant."""
    h, k, a = (np.asarray(joints[i], float) for i in LEGS[side])
    u, v = h - k, a - k
    return 180.0 - float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def gt_flexion(frames):
    """GT flexion per frame (one value for every camera) from the SMPL joints of the exported motion."""
    J = np.load(GT_NPZ)["joints"].astype(float)
    return {f: {s: knee_flexion(J[f], s) for s in LEGS} for f in frames}


def gait_phase(fps, start=60):
    """Phase in [0,1) from left heel strikes (max forward ankle-pelvis distance, Zeni 2008) and cycle index
    (1 = cycle after the first strike) for every motion frame >= start; returns (dict frame -> (phase, cycle), strikes)."""
    J = np.load(GT_NPZ)["joints"][start:].astype(float)
    d = J[-1, 0, :2] - J[0, 0, :2]
    s = (J[:, 7, :2] - J[:, 0, :2]) @ (d / np.linalg.norm(d))
    hs = find_peaks(s, distance=int(0.6 * fps), prominence=0.05)[0] + start
    out = {}
    for f in range(start, start + len(J)):
        c = int(np.searchsorted(hs, f, side="right"))
        out[f] = ((f - hs[c - 1]) / (hs[c] - hs[c - 1]) if 0 < c < len(hs) else np.nan, c)
    return out, hs.tolist()


def swap_legs(j):
    out = j.copy()
    for a, b in zip(LEGS["left"][1:], LEGS["right"][1:]):  # knees and ankles; hips belong to the pelvis
        out[[a, b]] = j[[b, a]]
    return out


def gt_identity_error(pred, gt):
    """True if the predicted legs match the GT better when swapped (after torso similarity alignment)."""
    src, dst = pred[TORSO], gt[TORSO]
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    p = (S * np.diag(D)).sum() / (a ** 2).sum() * (pred - src.mean(0)) @ R.T + dst.mean(0)
    legs = [i for s in LEGS.values() for i in s[1:]]
    err = lambda q: np.linalg.norm(q[legs] - gt[legs], axis=1).sum()
    return bool(err(swap_legs(p)) < err(p))


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def draw_legs(image, j3, gt_uv, K):
    """Predicted legs (camera-frame joints3d: left cyan, right orange) and GT legs (thin green)."""
    out = image.copy()
    if np.isfinite(j3).all():
        uv = j3[:, :2] / j3[:, 2:3] @ K[:2, :2].T + K[:2, 2]
        for s, idx in LEGS.items():
            colour = (0, 230, 255) if s == "left" else (255, 140, 0)
            cv2.polylines(out, [np.round(uv[list(idx)]).astype(np.int32)], False, colour, 3, cv2.LINE_AA)
            for i in idx:
                cv2.circle(out, tuple(np.round(uv[i]).astype(int)), 4, colour, -1, cv2.LINE_AA)
    for idx in LEGS.values():
        pts = np.round(gt_uv[list(idx)]).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 0), 1, cv2.LINE_AA)
        for q in pts:
            cv2.circle(out, tuple(q), 2, (0, 200, 0), -1, cv2.LINE_AA)
    return out


def person_crop(image, uv, margin=0.25):
    """Square crop around the points uv, zero-padded where it leaves the image."""
    lo, hi = uv.min(0), uv.max(0)
    side = (hi - lo).max() * (1 + 2 * margin)
    x0, y0 = np.round((lo + hi) / 2 - side / 2).astype(int)
    x1, y1 = x0 + int(side), y0 + int(side)
    pad = max(0, -x0, -y0, x1 - image.shape[1], y1 - image.shape[0])
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
