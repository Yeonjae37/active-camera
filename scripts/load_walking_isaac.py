
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Vt

ROOT = Path.home() / "Study/active-camera"
DATA_PATH = ROOT / "output/walking_data.npz"

# 1. Load exported walking data
with np.load(DATA_PATH, allow_pickle=False) as data:
    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int32)
    fps = int(data["fps"])

num_frames = len(vertices)

# 2. Get current Isaac Sim stage
stage = omni.usd.get_context().get_stage()

meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
up_axis = UsdGeom.GetStageUpAxis(stage)

print("Stage meters per unit:", meters_per_unit)
print("Stage up axis:", up_axis)

if str(up_axis) != "Z":
    raise RuntimeError(
        "This first version requires a Z-up USD stage."
    )

# SMPL coordinates are in meters.
scale = 1.0 / meters_per_unit

# 3. Hide old static SMPL human
old_human = stage.GetPrimAtPath("/World/smpl_neutral")

if old_human.IsValid():
    UsdGeom.Imageable(old_human).MakeInvisible()

# 4. Create animated mesh
mesh = UsdGeom.Mesh.Define(
    stage, "/World/WalkingHuman/Mesh"
)

mesh.CreateFaceVertexCountsAttr(
    [3] * len(faces)
)

mesh.CreateFaceVertexIndicesAttr(
    faces.reshape(-1).tolist()
)

mesh.CreateSubdivisionSchemeAttr().Set(
    UsdGeom.Tokens.none
)

mesh.CreateDoubleSidedAttr(True)

# 5. Configure animation timeline
stage.SetFramesPerSecond(fps)
stage.SetTimeCodesPerSecond(fps)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(num_frames - 1)

# 6. Animate mesh vertices over time
points_attr = mesh.CreatePointsAttr()

for frame in range(num_frames):
    points = np.ascontiguousarray(
        vertices[frame] * scale,
        dtype=np.float32,
    )

    points_attr.Set(
        Vt.Vec3fArray.FromNumpy(points),
        Usd.TimeCode(frame),
    )

print("Walking animation created!")
print("Frames:", num_frames)
print("FPS:", fps)
print("Duration:", num_frames / fps, "seconds")
print("Prim: /World/WalkingHuman/Mesh")
