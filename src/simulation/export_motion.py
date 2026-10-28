"""
AMASS motion -> SMPL mesh + 24 joints at 30 fps (npz for isaac/load_motion.py and the GT).
The motion starts at the origin (first-frame root xy) and the first frame's lowest vertex is put on z = 0.
python -m src.simulation.export_motion [--motion AMASS_poses.npz] [--out output/walking_data.npz]
"""

import argparse

import numpy as np
import smplx
import torch

from .common import ROOT

TARGET_FPS = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", default=ROOT / "data/amass/CMU/CMU/02/02_01_poses.npz")
    ap.add_argument("--out", default=ROOT / "output/motion_data.npz")
    ap.add_argument("--smpl", default=ROOT / "models/smpl/converted/SMPL_NEUTRAL.pkl")
    args = ap.parse_args()

    with np.load(args.motion) as d:
        poses, trans = d["poses"].astype(np.float32), d["trans"].astype(np.float32)
        betas, fps = d["betas"][:10].astype(np.float32), float(d["mocap_framerate"])
    idx = np.clip(np.round(np.arange(int(len(poses) / fps * TARGET_FPS)) * fps / TARGET_FPS).astype(int), 0, len(poses) - 1)
    model = smplx.SMPL(model_path=str(args.smpl), gender="neutral", num_betas=10, batch_size=len(idx))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    with torch.no_grad():
        out = model(betas=t(np.repeat(betas[None], len(idx), 0)), global_orient=t(poses[idx, :3]),
                    body_pose=t(np.c_[poses[idx, 3:66], np.zeros((len(idx), 6), np.float32)]), transl=t(trans[idx]))
    verts, joints = out.vertices.numpy(), out.joints[:, :24].numpy()
    offset = np.r_[trans[idx[0], :2], verts[0, :, 2].min()]
    verts, joints = verts - offset, joints - offset
    np.savez_compressed(args.out, vertices=verts, joints=joints, faces=model.faces.astype(np.int32), fps=np.array(TARGET_FPS))
    print(f"{args.motion}: {len(idx)} frames -> {args.out}")


if __name__ == "__main__":
    main()
