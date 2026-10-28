"""
for Isaac Sim Script Editor
load an exported motion (export_motion.py) as a time-sampled mesh at /World/WalkingHuman/Mesh (Z-up stage) and set the timeline to its fps.
"""

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Vt

DATA_PATH = "/home/yeonjae-jeong/Study/active-camera/output/walking_data.npz"

with np.load(DATA_PATH) as d:
    vertices, faces, fps = d["vertices"].astype(np.float32), d["faces"].astype(np.int32), int(d["fps"])

stage = omni.usd.get_context().get_stage()
if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
    raise RuntimeError("Z-up stage required")
scale = 1.0 / UsdGeom.GetStageMetersPerUnit(stage)  # SMPL is in metres

old = stage.GetPrimAtPath("/World/smpl_neutral")
if old.IsValid():
    UsdGeom.Imageable(old).MakeInvisible()

mesh = UsdGeom.Mesh.Define(stage, "/World/WalkingHuman/Mesh")
mesh.CreateFaceVertexCountsAttr([3] * len(faces))
mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)
stage.SetFramesPerSecond(fps)
stage.SetTimeCodesPerSecond(fps)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(len(vertices) - 1)
points = mesh.CreatePointsAttr()
for f, v in enumerate(vertices):
    points.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v * scale)), Usd.TimeCode(f))
print(f"Loaded {len(vertices)} frames at {fps} fps -> /World/WalkingHuman/Mesh")
