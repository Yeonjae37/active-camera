
import torch
import smplx
import trimesh
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/smpl/converted/SMPL_NEUTRAL.pkl"
OUTPUT_PATH = ROOT / "output/mesh/smpl_neutral.obj"

# Load SMPL model
model = smplx.SMPL(
    model_path=str(MODEL_PATH),
    gender="neutral",
    num_betas=10,
    batch_size=1,
)

# Neutral body shape and default pose
betas = torch.zeros(1, 10)
body_pose = torch.zeros(1, 69)
global_orient = torch.zeros(1, 3)
transl = torch.zeros(1, 3)

with torch.no_grad():
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
    )

vertices = output.vertices[0].cpu().numpy()
joints = output.joints[0].cpu().numpy()
faces = model.faces

# Export mesh
mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False,
)

mesh.export(OUTPUT_PATH)

print("SMPL mesh created!")
print("Vertices:", vertices.shape)
print("Joints:", joints.shape)
print("Faces:", faces.shape)
print("Saved to:", OUTPUT_PATH)
