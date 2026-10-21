
from pathlib import Path

import numpy as np
import torch
import smplx

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/smpl/converted/SMPL_NEUTRAL.pkl"
MOTION_PATH = ROOT / "data/amass/CMU/CMU/02/02_01_poses.npz"
OUTPUT_PATH = ROOT / "output/walking_data.npz"

TARGET_FPS = 30
DURATION = 3.0  # seconds

# 1. Load AMASS
with np.load(MOTION_PATH, allow_pickle=False) as data:
    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"][:10].astype(np.float32)
    source_fps = float(data["mocap_framerate"])

print("Motion:", MOTION_PATH.name)
print("Original frames:", len(poses))
print("Original FPS:", source_fps)

# 2. Sample frames at 30 FPS
frame_count = min(
    int(DURATION * TARGET_FPS),
    int(len(poses) / source_fps * TARGET_FPS),
)

if frame_count < 2:
    raise RuntimeError("Motion sequence is too short")

sample_indices = np.round(
    np.arange(frame_count) * source_fps / TARGET_FPS
).astype(int)

sample_indices = np.clip(sample_indices, 0, len(poses) - 1)

# 3. Load SMPL
model = smplx.SMPL(
    model_path=str(MODEL_PATH),
    gender="neutral",
    num_betas=10,
    batch_size=1,
)

betas_tensor = torch.from_numpy(betas)[None, :]

all_vertices = []
all_joints = []

# 4. Generate SMPL mesh per frame
with torch.no_grad():
    for frame_idx, source_idx in enumerate(sample_indices):
        pose = poses[source_idx]

        result = model(
            betas=betas_tensor,
            global_orient=torch.from_numpy(
                pose[:3].copy()
            )[None, :],
            body_pose=torch.from_numpy(
                np.concatenate([
                    pose[3:66],
                    np.zeros(6, dtype=np.float32),
                ])
            )[None, :],
            transl=torch.from_numpy(
                trans[source_idx].copy()
            )[None, :],
        )

        all_vertices.append(
            result.vertices[0].cpu().numpy()
        )

        all_joints.append(
            result.joints[0, :24].cpu().numpy()
        )

vertices = np.stack(all_vertices).astype(np.float32)
joints = np.stack(all_joints).astype(np.float32)

# 5. Start motion near the center of the stage
# Keep the original height and movement along the ground.
initial_xy = trans[sample_indices[0], :2].copy()

vertices[:, :, :2] -= initial_xy
joints[:, :, :2] -= initial_xy

# 6. Put the lowest vertex of the FIRST frame at Z=0
ground_offset = float(vertices[0, :, 2].min())

vertices[:, :, 2] -= ground_offset
joints[:, :, 2] -= ground_offset

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH,
    vertices=vertices,
    joints=joints,
    faces=model.faces.astype(np.int32),
    fps=np.array(TARGET_FPS),
)

print("Export completed!")
print("Vertices:", vertices.shape)
print("Joints:", joints.shape)
print("Faces:", model.faces.shape)
print("Saved:", OUTPUT_PATH)
