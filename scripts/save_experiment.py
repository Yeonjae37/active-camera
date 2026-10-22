
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.usd

from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path.home() / "Study/active-camera"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = ROOT / "output/experiment_backups" / STAMP

CAMERA_PATH = (
    "/World/pan_tilt/Geometry/base_link/pan_link/"
    "tilt_link/camera_base_link/camera_link/"
    "camera_depth_frame/camera_color_frame/"
    "camera_color_optical_frame/FemtoBolt_RGB"
)

PAN_BASE_PATH = "/World/pan_tilt/Geometry/base_link"
HUMAN_PATH = "/World/WalkingHuman"
MESH_PATH = HUMAN_PATH + "/Mesh"


def matrix_to_list(matrix):
    return [
        [float(matrix[r][c]) for c in range(4)]
        for r in range(4)
    ]


def world_pose(stage, path, time_code):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        return None

    matrix = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode(time_code)
    )

    return {
        "path": path,
        "position_xyz_m": [
            float(v) for v in matrix.ExtractTranslation()
        ],
        "local_to_world_matrix": matrix_to_list(matrix),
    }


def read_joint(stage, name):
    path = f"/World/pan_tilt/Physics/{name}"
    joint = UsdPhysics.RevoluteJoint.Get(stage, path)

    if not joint:
        return {"path": path, "found": False}

    drive = UsdPhysics.DriveAPI.Get(
        joint.GetPrim(), "angular"
    )

    target = None

    if drive:
        value = drive.GetTargetPositionAttr().Get()
        if value is not None:
            target = float(value)

    return {
        "path": path,
        "found": True,
        "target_angle_deg": target,
        "lower_limit_deg": joint.GetLowerLimitAttr().Get(),
        "upper_limit_deg": joint.GetUpperLimitAttr().Get(),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_if_exists(src, dst):
    if not src.exists():
        print("Not found:", src)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

    print("Copied:", src)
    return True


stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError("No USD stage is open")

camera = UsdGeom.Camera.Get(stage, CAMERA_PATH)

if not camera:
    raise RuntimeError("FemtoBolt_RGB camera not found")

BACKUP.mkdir(parents=True, exist_ok=True)

# Export the current composed USD scene.
scene_path = BACKUP / "scene/scene_snapshot.usdc"
scene_path.parent.mkdir(parents=True, exist_ok=True)

if not stage.Export(str(scene_path)):
    raise RuntimeError("USD scene export failed")

print("Scene exported:", scene_path)

# Save current camera parameters.
hap = float(camera.GetHorizontalApertureAttr().Get())
vap = float(camera.GetVerticalApertureAttr().Get())
focal = float(camera.GetFocalLengthAttr().Get())

width, height = 1920, 1080

fx = focal / hap * width
fy = focal / vap * height

camera_info = {
    "camera_path": CAMERA_PATH,
    "resolution": [width, height],
    "intrinsics": {
        "fx": fx,
        "fy": fy,
        "cx": width / 2,
        "cy": height / 2,
        "horizontal_aperture": hap,
        "vertical_aperture": vap,
        "focal_length": focal,
    },
    "distortion_model": "ideal_pinhole",
    "distortion_coefficients": None,
    "camera_pose": world_pose(
        stage, CAMERA_PATH, stage.GetStartTimeCode()
    ),
    "pan_tilt_base_pose": world_pose(
        stage, PAN_BASE_PATH, stage.GetStartTimeCode()
    ),
    "pan_joint": read_joint(stage, "pan_joint"),
    "tilt_joint": read_joint(stage, "tilt_joint"),
}

save_json(BACKUP / "metadata/camera.json", camera_info)

# Save scene and animation information.
mesh = UsdGeom.Mesh.Get(stage, MESH_PATH)
samples = mesh.GetPointsAttr().GetTimeSamples() if mesh else []

scene_info = {
    "original_stage": stage.GetRootLayer().identifier,
    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    "time_codes_per_second": stage.GetTimeCodesPerSecond(),
    "start_time_code": stage.GetStartTimeCode(),
    "end_time_code": stage.GetEndTimeCode(),
    "walking_mesh_path": MESH_PATH,
    "walking_animation_samples": len(samples),
    "walking_human_pose": world_pose(
        stage, HUMAN_PATH, stage.GetStartTimeCode()
    ),
}

save_json(BACKUP / "metadata/scene.json", scene_info)

# Copy the SMPL / AMASS export.
copy_if_exists(
    ROOT / "output/walking_data.npz",
    BACKUP / "data/walking_data.npz"
)

# Copy all recorded capture sessions without guessing their modes.
capture_root = ROOT / "output/walking_capture"

if capture_root.exists():
    for folder in sorted(capture_root.iterdir()):
        if folder.is_dir():
            copy_if_exists(
                folder,
                BACKUP / "captures" / folder.name
            )

# Copy project scripts.
if (ROOT / "scripts").exists():
    shutil.copytree(
        ROOT / "scripts",
        BACKUP / "scripts",
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc"
        ),
    )

# Record the source motion and SMPL paths.
manifest = {
    "backup_time": STAMP,
    "backup_directory": str(BACKUP),
    "note": (
        "This snapshot records the current scene state. "
        "Historical camera motion cannot be recovered "
        "from RGB images alone."
    ),
    "motion_source_path": (
        "data/amass/CMU/CMU/136/136_24_poses.npz"
    ),
    "smpl_model_path": (
        "models/smpl/converted/SMPL_NEUTRAL.pkl"
    ),
    "capture_directories": [
        p.name for p in sorted(capture_root.iterdir())
        if p.is_dir()
    ] if capture_root.exists() else [],
}

save_json(BACKUP / "metadata/manifest.json", manifest)

print("\nExperiment backup completed!")
print("Backup:", BACKUP)
