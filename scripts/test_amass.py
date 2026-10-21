
import numpy as np
import torch
import smplx
import trimesh
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/smpl/converted/SMPL_NEUTRAL.pkl"
MOTION_PATH = ROOT / "data/amass/CMU/CMU/01/01_01_poses.npz"
OUTPUT_DIR = ROOT / "output/mesh/amass_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load AMASS motion
with np.load(MOTION_PATH, allow_pickle=False) as data:
    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"][:10].astype(np.float32)
    fps = float(data["mocap_framerate"])

print("AMASS:", poses.shape, trans.shape)
print("FPS:", fps)

# 2. Convert SMPL+H body pose to SMPL pose
T = len(poses)

smpl_poses = np.zeros((T, 72), dtype=np.float32)
smpl_poses[:, :66] = poses[:, :66]

# The last two SMPL hand joints remain neutral
# Note: SMPL+H and SMPL joint mapping should be verified.

# 3. Load SMPL
model = smplx.SMPL(
    model_path=str(MODEL_PATH),
    gender="neutral",
    num_betas=10,
    batch_size=1,
)

# 4. Export a small sample of frames
sample_indices = np.linspace(
    0, min(T - 1, int(fps * 2)),
    num=20,
    dtype=int,
)

for i in sample_indices:
    with torch.no_grad():
        result = model(
            betas=torch.from_numpy(betas).unsqueeze(0),
            global_orient=torch.from_numpy(
                smpl_poses[i, :3]
            ).unsqueeze(0),
            body_pose=torch.from_numpy(
                smpl_poses[i, 3:]
            ).unsqueeze(0),
            transl=torch.from_numpy(
                trans[i]
            ).unsqueeze(0),
        )

    vertices = result.vertices[0].numpy()
    joints = result.joints[0].numpy()

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=model.faces,
        process=False,
    )

    mesh.export(OUTPUT_DIR / f"frame_{i:05d}.obj")

    np.save(
        OUTPUT_DIR / f"joints_{i:05d}.npy",
        joints,
    )

print("Export complete:", OUTPUT_DIR)
