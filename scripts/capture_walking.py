
import asyncio
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.usd
import omni.timeline

from pxr import Usd, UsdGeom, UsdPhysics, Gf

from omni.kit.viewport.utility import (
    get_active_viewport,
    capture_viewport_to_file,
    next_viewport_frame_async,
)

ROOT = Path.home() / "Study/active-camera"
DATA_PATH = ROOT / "output/walking_data.npz"

# Change this to "manual_tracking" for the second recording.
MODE = "manual_tracking"
#MODE = "fixed"

DURATION = 10.0
NUM_FRAMES = 30

WIDTH = 1920
HEIGHT = 1080

CAMERA_PATH = (
    "/World/pan_tilt/Geometry/base_link/pan_link/"
    "tilt_link/camera_base_link/camera_link/"
    "camera_depth_frame/camera_color_frame/"
    "camera_color_optical_frame/FemtoBolt_RGB"
)

HUMAN_PATH = "/World/WalkingHuman/Mesh"
ARTICULATION_PATH = "/World/pan_tilt/Geometry/base_link"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = (
    ROOT / "output/experiments"
    / MODE / RUN_ID
)


def matrix_to_numpy(matrix):
    return np.array(
        [
            [float(matrix[r][c]) for c in range(4)]
            for r in range(4)
        ],
        dtype=np.float64,
    )


def get_world_matrix(prim, time_sec, stage):
    time_code = Usd.TimeCode(
        time_sec * stage.GetTimeCodesPerSecond()
    )

    matrix = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(time_code)

    return matrix_to_numpy(matrix)


def transform_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)

    homogeneous = np.concatenate(
        [
            points,
            np.ones((len(points), 1)),
        ],
        axis=1,
    )

    # USD matrices use row-vector convention.
    return (homogeneous @ matrix)[:, :3]


def get_joint_target(stage, joint_name):
    path = f"/World/pan_tilt/Physics/{joint_name}"

    joint = UsdPhysics.RevoluteJoint.Get(stage, path)

    if not joint:
        return None

    drive = UsdPhysics.DriveAPI.Get(
        joint.GetPrim(), "angular"
    )

    if not drive:
        return None

    value = drive.GetTargetPositionAttr().Get()

    return float(value) if value is not None else None


def get_intrinsics(camera):
    hap = float(camera.GetHorizontalApertureAttr().Get())
    vap = float(camera.GetVerticalApertureAttr().Get())
    focal = float(camera.GetFocalLengthAttr().Get())

    fx = focal / hap * WIDTH
    fy = focal / vap * HEIGHT

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "fx": fx,
        "fy": fy,
        "cx": WIDTH / 2.0,
        "cy": HEIGHT / 2.0,
        "focal_length": focal,
        "horizontal_aperture": hap,
        "vertical_aperture": vap,
        "model": "ideal_pinhole",
    }


def interpolate_joints(joints, time_sec, fps):
    frame_position = np.clip(
        time_sec * fps,
        0.0,
        len(joints) - 1.0,
    )

    i0 = int(math.floor(frame_position))
    i1 = min(i0 + 1, len(joints) - 1)

    alpha = frame_position - i0

    return (
        (1.0 - alpha) * joints[i0]
        + alpha * joints[i1]
    )


def read_actual_joint_angles(sensor):
    result = {
        "pan_actual_deg": None,
        "tilt_actual_deg": None,
        "joint_sensor_time_sec": None,
    }

    if sensor is None:
        return result

    try:
        reading = sensor.get_sensor_reading()

        if not reading.is_valid:
            return result

        result["joint_sensor_time_sec"] = float(reading.time)

        for name, position in zip(
            reading.dof_names,
            reading.positions,
        ):
            if name == "pan_joint":
                result["pan_actual_deg"] = math.degrees(
                    float(position)
                )

            elif name == "tilt_joint":
                result["tilt_actual_deg"] = math.degrees(
                    float(position)
                )

    except Exception as exc:
        print("Joint sensor reading unavailable:", exc)

    return result


async def capture_walking():
    if MODE not in ("fixed", "manual_tracking"):
        raise ValueError("Invalid capture mode")

    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    timeline = omni.timeline.get_timeline_interface()

    if viewport is None:
        raise RuntimeError("Active viewport not found")

    if str(viewport.camera_path) != CAMERA_PATH:
        raise RuntimeError(
            "Select FemtoBolt_RGB in the active viewport "
            "before recording."
        )

    human_prim = stage.GetPrimAtPath(HUMAN_PATH)
    camera = UsdGeom.Camera.Get(stage, CAMERA_PATH)

    if not human_prim.IsValid() or not camera:
        raise RuntimeError("Human or camera not found")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        joints = data["joints"].astype(np.float64)
        fps = int(data["fps"])

    stage_fps = stage.GetTimeCodesPerSecond()

    if abs(stage_fps - fps) > 1e-6:
        raise RuntimeError(
            f"FPS mismatch: stage={stage_fps}, data={fps}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    end_time = min(
        DURATION,
        timeline.get_end_time() - 0.1,
        (len(joints) - 1) / fps,
    )

    if end_time <= 0:
        raise RuntimeError("Invalid recording duration")

    target_times = [
        (i + 0.5) * end_time / NUM_FRAMES
        for i in range(NUM_FRAMES)
    ]

    print("Mode:", MODE)
    print("Camera:", CAMERA_PATH)
    print("Capture duration:", end_time)
    print("Target frames:", NUM_FRAMES)
    print("Output:", OUTPUT_DIR)

    sensor = None

    try:
        from isaacsim.sensors.experimental.physics import (
            JointStateSensor,
        )

        sensor = JointStateSensor(
            ARTICULATION_PATH
        )

        print("Joint state sensor initialized")

    except Exception as exc:
        print("Joint sensor unavailable:", exc)

    records = []
    joints_gt = []
    camera_poses = []
    human_poses = []

    # Only timeline state is changed; no mesh animation is edited.
    timeline.stop()
    timeline.set_looping(False)
    timeline.set_current_time(0.0)

    try:
        timeline.play()

        await next_viewport_frame_async(
            viewport, n_frames=1
        )

        for i, target_time in enumerate(target_times):

            # Advance the walking animation to the target time.
            while timeline.get_current_time() < target_time:
                if not timeline.is_playing():
                    raise RuntimeError(
                        "Timeline stopped at "
                        f"{timeline.get_current_time():.3f}s"
                    )

                await next_viewport_frame_async(
                    viewport, n_frames=1
                )

            timeline.pause()

            # Let the viewport settle on the paused scene.
            await next_viewport_frame_async(
                viewport, n_frames=2
            )

            actual_time = timeline.get_current_time()

            # Record camera and human state at the paused time.
            camera_matrix = get_world_matrix(
                camera.GetPrim(),
                actual_time,
                stage,
            )

            human_matrix = get_world_matrix(
                human_prim,
                actual_time,
                stage,
            )

            local_joints = interpolate_joints(
                joints,
                actual_time,
                fps,
            )

            world_joints = transform_points(
                human_matrix,
                local_joints,
            )

            actual_angles = read_actual_joint_angles(sensor)

            pan_target = get_joint_target(
                stage, "pan_joint"
            )

            tilt_target = get_joint_target(
                stage, "tilt_joint"
            )

            image_name = f"{i:06d}.png"
            image_path = OUTPUT_DIR / image_name

            # Keep the timeline paused during PNG writing.
            capture = capture_viewport_to_file(
                viewport,
                file_path=str(image_path),
            )

            await capture.wait_for_result(
                completion_frames=60
            )

            joints_gt.append(world_joints)
            camera_poses.append(camera_matrix)
            human_poses.append(human_matrix)

            records.append({
                "frame": i,
                "image": image_name,
                "target_time_sec": float(target_time),
                "actual_time_sec": float(actual_time),
                "pan_target_deg": pan_target,
                "tilt_target_deg": tilt_target,
                **actual_angles,
            })

            print(
                f"[{MODE}] "
                f"{i + 1:02d}/{NUM_FRAMES} "
                f"| t={actual_time:.3f}s "
                f"| pan={pan_target} "
                f"| tilt={tilt_target}",
                flush=True,
            )

            if i < NUM_FRAMES - 1:
                timeline.play()

                await next_viewport_frame_async(
                    viewport, n_frames=1
                )

        print("Capture completed!")

    finally:
        timeline.pause()

        # Save all completed frames even after an interruption.
        if records:
            np.save(
                OUTPUT_DIR / "joints_gt.npy",
                np.stack(joints_gt),
            )

            np.save(
                OUTPUT_DIR / "camera_poses.npy",
                np.stack(camera_poses),
            )

            np.save(
                OUTPUT_DIR / "human_poses.npy",
                np.stack(human_poses),
            )

            with open(
                OUTPUT_DIR / "frames.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    records,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            with open(
                OUTPUT_DIR / "timestamps.csv",
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(records[0].keys()),
                )
                writer.writeheader()
                writer.writerows(records)

        metadata = {
            "mode": MODE,
            "camera_path": CAMERA_PATH,
            "human_path": HUMAN_PATH,
            "duration_requested_sec": DURATION,
            "duration_used_sec": end_time,
            "target_frames": NUM_FRAMES,
            "captured_frames": len(records),
            "animation_fps": fps,
            "stage_time_codes_per_second": stage_fps,
            "meters_per_unit":
                UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "intrinsics": get_intrinsics(camera),
            "camera_matrix_convention":
                "USD row-vector, camera local to world",
            "joint_gt_reference":
                "SMPL 24 joints interpolated by timeline time",
            "notes": (
                "Viewport capture with pause/resume. "
                "Actual frame times may not be uniform. "
                "Actual joint readings are null if unavailable. "
                "Manual tracking uses user-controlled pan/tilt."
            ),
        }

        with open(
            OUTPUT_DIR / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("Saved frames:", len(records))
        print("Data directory:", OUTPUT_DIR)


# Retain the task reference in the Script Editor session.
capture_task = asyncio.ensure_future(capture_walking())
