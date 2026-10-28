"""
for Isaac Sim Script Editor
render camera sweeps over a walking sequence (images + GT metadata only).

Load the motion first with load_motion.py, set PRESET, run. Kinds:
  pan_tilt : mounted camera position, OFFSETS x OFFSETS pan/tilt offsets around the human centre.
  rail     : virtual rail through the mounted camera along the horizontal walking direction; camera at
             RAIL_M, auto-aimed at the human centre.
  orbit    : horizontal circle of ORBIT_RADIUS_M around the human centre at the mounted camera height,
             ORBIT_AZ_DEG from the walking direction (0 = front, 90 = person's right); distance fixed.
"""

import asyncio
import json
import math
import os

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdShade, Vt

PRESET = "rail"
PRESETS = {  # CMU 02_01: steady walking from ~frame 60, left heel strikes 79, 126, 172, 221
    "pan_tilt": dict(kind="pan_tilt", sequence_id="cmu_02_01_walk15", out="01_pantilt_framing",
                     frames=list(range(79, 169, 6)), offsets=[-15, -10, -5, 0, 5, 10, 15]),
    "rail": dict(kind="rail", sequence_id="cmu_02_01_rail_3cycles", out="02_rail_viewpoint",
                 frames=list(range(79, 221)), rail_m=[round(0.1 * i, 1) for i in range(-6, 7)]),
    "orbit": dict(kind="orbit", sequence_id="cmu_02_01_orbit_3cycles", out="03_orbit_azimuth",
                  frames=list(range(79, 221, 3)), orbit_az_deg=list(range(30, 151, 15)), orbit_radius_m=2.75),
}
P = PRESETS[PRESET]
KIND, FRAMES = P["kind"], P["frames"]
OFFSETS = P.get("offsets", [0])
RAIL_M = P.get("rail_m", [0.0])
ORBIT_AZ_DEG = P.get("orbit_az_deg", [])
ORBIT_RADIUS_M = P.get("orbit_radius_m")

ROOT = os.path.expanduser("~/Study/active-camera")
GT_NPZ = f"{ROOT}/output/walking_data.npz"  # must be the motion loaded by load_motion.py
OUTPUT_DIR = f"{ROOT}/output/{P['out']}/capture"
SOURCE_CAMERA = ("/World/pan_tilt/Geometry/base_link/pan_link/tilt_link/camera_base_link/camera_link/"
                 "camera_depth_frame/camera_color_frame/camera_color_optical_frame/FemtoBolt_RGB")
HUMAN_MESH = "/World/WalkingHuman/Mesh"
CAMERA = "/World/FramingSweepCamera"
PROXY = "/World/FramingSweepHuman"
WIDTH, HEIGHT = 960, 540
RT_SUBFRAMES = 8  # extra RTX subframes so the previous pose does not ghost into the image
RESUME = True
MIN_WALK_SPEED_M_S = 0.3  # refuse standing frames
CHECK_JOINTS, CHECK_TOL_M = [0, 3, 6, 9], 0.3  # pelvis, spine1-3: median |rendered - GT depth|

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
session = stage.GetSessionLayer()
MPU = UsdGeom.GetStageMetersPerUnit(stage)
UP = np.array([0.0, 0.0, 1.0]) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else np.array([0.0, 1.0, 0.0])


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def flat(v):
    return np.asarray(v, float) - UP * np.dot(v, UP)


def rotate(v, axis, deg):
    a, k = math.radians(deg), unit(axis)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


def camera_pose(eye, target, pan, tilt):
    f = rotate(unit(target - eye), UP, pan)
    f = unit(rotate(f, np.cross(f, UP), tilt))
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*(eye + f).tolist()), Gf.Vec3d(*UP.tolist()))
    return view.GetInverse()


def as_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def mesh_xform(frame):
    return UsdGeom.XformCache(Usd.TimeCode(frame)).GetLocalToWorldTransform(stage.GetPrimAtPath(HUMAN_MESH))


def to_world(local_m, frame):
    """Exported points (m, motion frame) -> world, USD row-vector convention."""
    return (np.c_[local_m / MPU, np.ones(len(local_m))] @ as_np(mesh_xform(frame)))[:, :3]


def project(pose, K, p):
    """World point -> (u, v, depth) for a USD camera (looks down -Z, +Y up)."""
    c = np.r_[p, 1.0] @ np.linalg.inv(as_np(pose))
    z = -c[2]
    return int(round(K[0][0] * c[0] / z + K[0][2])), int(round(-K[1][1] * c[1] / z + K[1][2])), z


def depth_gaps(depth, pose, K, joints):
    """Rendered depth - joint depth (m) per joint; None if behind the camera or off-image."""
    out = []
    for p in joints:
        u, v, z = project(pose, K, p)
        ok = z > 0 and 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]
        out.append((float(depth[v, u]) - z) * MPU if ok else None)
    return out


def view_geometry(eye, joints, target, walk):
    """Azimuth (0 = front, +90 = person's right) from the hips and from the walking direction."""
    left = unit(flat(joints[1] - joints[2]))
    fwd = unit(np.cross(left, UP))
    to_cam = eye - joints[0]
    h = flat(to_cam)
    return {
        "view_azimuth_deg": math.degrees(math.atan2(np.dot(h, -left), np.dot(h, fwd))),
        "view_azimuth_walk_deg": math.degrees(math.atan2(np.dot(h, np.cross(walk, UP)), np.dot(h, walk))),
        "view_elevation_deg": math.degrees(math.atan2(np.dot(to_cam, UP), np.linalg.norm(h))),
        "camera_to_pelvis_horizontal_m": float(np.linalg.norm(h) * MPU),
        "camera_to_pelvis_m": float(np.linalg.norm(to_cam) * MPU),
        "camera_tilt_deg": math.degrees(math.asin(np.clip(np.dot(unit(target - eye), UP), -1, 1))),
    }


def file_name(frame, pan, tilt, rail_m, az):
    if KIND == "orbit":
        return f"frame_{frame:03d}_az_{az:03d}.png"
    if KIND == "rail":
        return f"frame_{frame:03d}_rail_{rail_m:+.2f}m.png"
    return f"frame_{frame:03d}_pan_{pan:+03d}_tilt_{tilt:+03d}.png"


def setup_proxy(faces):
    src = stage.GetPrimAtPath(HUMAN_MESH)
    with Usd.EditContext(stage, session):
        proxy = UsdGeom.Mesh.Define(stage, PROXY)
        proxy.CreateFaceVertexCountsAttr([3] * len(faces))
        proxy.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        proxy.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        proxy.CreateDoubleSidedAttr(True)
        xf = UsdGeom.Xformable(proxy.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp()
        material, _ = UsdShade.MaterialBindingAPI(src).ComputeBoundMaterial()
        if material:
            UsdShade.MaterialBindingAPI.Apply(proxy.GetPrim()).Bind(material)
        UsdGeom.Imageable(proxy.GetPrim()).MakeVisible()
        UsdGeom.Imageable(src).MakeInvisible()
    return proxy


def set_proxy(proxy, verts, frame):
    pts = np.ascontiguousarray(verts[frame] / MPU, dtype=np.float32)
    with Usd.EditContext(stage, session):  # default values: the render cannot pick up another frame
        proxy.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
        proxy.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())]))
        UsdGeom.Xformable(proxy.GetPrim()).GetOrderedXformOps()[0].Set(mesh_xform(frame))


def restore_human():
    with Usd.EditContext(stage, session):
        UsdGeom.Imageable(stage.GetPrimAtPath(HUMAN_MESH)).MakeVisible()
        if stage.GetPrimAtPath(PROXY).IsValid():
            UsdGeom.Imageable(stage.GetPrimAtPath(PROXY)).MakeInvisible()


def mesh_sync_error(verts, frame):
    """Max |loaded mesh points - exported vertices| at this frame: renders and GT share one motion."""
    pts = UsdGeom.Mesh.Get(stage, HUMAN_MESH).GetPointsAttr().Get(Usd.TimeCode(frame))
    if pts is None or len(pts) != len(verts[frame]):
        return None
    return float(np.abs(np.asarray(pts, float) - verts[frame] / MPU).max())


def make_camera(source):
    with Usd.EditContext(stage, session):
        cam = UsdGeom.Camera.Define(stage, CAMERA)
        cam.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
        for name in ("FocalLength", "HorizontalAperture", "VerticalAperture", "ClippingRange"):
            getattr(cam, f"Get{name}Attr")().Set(getattr(source, f"Get{name}Attr")().Get())
        xf = UsdGeom.Xformable(cam.GetPrim())
        existing = cam.GetPrim().GetAttribute("xformOp:transform")
        if existing:
            op = UsdGeom.XformOp(existing)
            xf.SetXformOpOrder([op])
        else:
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
    f, h, v = (float(getattr(source, f"Get{n}Attr")().Get()) for n in ("FocalLength", "HorizontalAperture", "VerticalAperture"))
    K = [[WIDTH * f / h, 0.0, WIDTH / 2], [0.0, HEIGHT * f / v, HEIGHT / 2], [0.0, 0.0, 1.0]]
    return op, K


async def run():
    source = UsdGeom.Camera.Get(stage, SOURCE_CAMERA)
    if not source:
        raise RuntimeError(f"Source camera not found: {SOURCE_CAMERA}")
    with np.load(GT_NPZ) as d:
        joints, verts, faces, fps = d["joints"].astype(float), d["vertices"].astype(float), d["faces"].astype(np.int32), float(d["fps"])
    if abs(stage.GetTimeCodesPerSecond() - fps) > 1e-6:
        raise RuntimeError(f"FPS mismatch: stage {stage.GetTimeCodesPerSecond()}, data {fps}")
    frames = sorted(FRAMES)
    speed = {f: float(np.linalg.norm(joints[min(f + 1, len(joints) - 1), 0, :2] - joints[max(f - 1, 0), 0, :2]) * fps / 2)
             for f in frames}
    standing = {f: round(s, 2) for f, s in speed.items() if s < MIN_WALK_SPEED_M_S}
    if standing:
        raise RuntimeError(f"Frames without walking motion (pelvis m/s): {standing}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = f"{OUTPUT_DIR}/metadata.json"
    timeline.pause()
    eye = np.array(UsdGeom.XformCache(Usd.TimeCode(frames[0])).GetLocalToWorldTransform(source.GetPrim()).ExtractTranslation())
    gt_world = lambda f: to_world(joints[f], f)
    walk = unit(flat(gt_world(frames[-1])[0] - gt_world(frames[0])[0]))
    records, frame_info = [], {}
    if RESUME and os.path.isfile(meta_path):
        prev = json.load(open(meta_path))
        if (np.abs(np.asarray(prev["camera_position_world"]) - eye).max() > 1e-6
                or prev.get("rail", {}).get("positions_m", RAIL_M) != RAIL_M
                or prev.get("orbit", {}).get("azimuths_deg", ORBIT_AZ_DEG) != ORBIT_AZ_DEG):
            raise RuntimeError("Camera or views changed since the previous run; use a new sequence_id")
        records, frame_info = prev["captures"], {int(k): v for k, v in prev["frame_info"].items()}

    pose_op, K = make_camera(source)

    def views(center):
        if KIND == "orbit":
            base = flat(center) + UP * np.dot(eye, UP)
            right = np.cross(walk, UP)
            return [(base + ORBIT_RADIUS_M / MPU * (math.cos(math.radians(a)) * walk + math.sin(math.radians(a)) * right),
                     0, 0, None, a) for a in ORBIT_AZ_DEG]
        return [(eye + walk * d / MPU, p, t, d, None) for d in RAIL_M for p in OFFSETS for t in OFFSETS]

    def write_metadata():
        json.dump({
            "resolution": [WIDTH, HEIGHT], "source_camera": SOURCE_CAMERA, "sequence_id": P["sequence_id"],
            "motion_data": GT_NPZ, "fps": fps, "experiment": KIND, "preset": PRESET,
            "camera_position_world": eye.tolist(), "offsets_deg": OFFSETS,
            "rail": {"origin_world": eye.tolist(), "direction_world": walk.tolist(), "positions_m": RAIL_M},
            "orbit": {"azimuths_deg": ORBIT_AZ_DEG, "radius_m": ORBIT_RADIUS_M},
            "rt_subframes": RT_SUBFRAMES, "frames": sorted({r["frame"] for r in records}),
            "frame_info": {str(k): v for k, v in sorted(frame_info.items())}, "captures": records,
        }, open(meta_path, "w"), indent=2)

    product = rep.create.render_product(CAMERA, (WIDTH, HEIGHT))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb.attach([product])
    depth.attach([product])
    proxy = setup_proxy(faces)

    async def render():
        for _ in range(5):  # the first steps after creating a render product can be empty (warm-up)
            await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=True)
            img = np.asarray(rgb.get_data())
            if img.size and img.ndim == 3:
                return img
        raise RuntimeError("Empty RGB capture")

    try:
        for frame in frames:
            done = [r["file"] for r in records if r["frame"] == frame]
            if RESUME and done and len(done) == len(views(np.zeros(3))) and all(os.path.isfile(f"{OUTPUT_DIR}/{f}") for f in done):
                print(f"Frame {frame}: already captured")
                continue
            records = [r for r in records if r["frame"] != frame]
            t = frame / stage.GetTimeCodesPerSecond()
            timeline.set_current_time(t)
            await omni.kit.app.get_app().next_update_async()
            set_proxy(proxy, verts, frame)
            v_world = to_world(verts[frame], frame)
            center = (v_world.min(0) + v_world.max(0)) / 2  # bounding-box centre
            gt = gt_world(frame)
            sync = mesh_sync_error(verts, frame)
            if sync is None or sync > 1e-4:
                raise RuntimeError(f"Frame {frame}: loaded mesh != {GT_NPZ} (err {sync}); run load_motion.py")
            checks = {}
            for cam_eye, pan, tilt, rail_m, az in views(center):
                pose = camera_pose(cam_eye, center, pan, tilt)
                with Usd.EditContext(stage, session):
                    pose_op.Set(pose)
                img = await render()
                gaps = None
                if pan == 0 and tilt == 0:
                    gaps = depth_gaps(np.asarray(depth.get_data()), pose, K, gt)
                    torso = [abs(gaps[j]) for j in CHECK_JOINTS if gaps[j] is not None]
                    err = float(np.median(torso)) if torso else None
                    if err is None or err > CHECK_TOL_M:
                        raise RuntimeError(f"Frame {frame}: rendered body does not match GT (torso gaps {torso})")
                    checks[f"az{az:03d}" if KIND == "orbit" else f"{rail_m:+.2f}"] = err
                name = file_name(frame, pan, tilt, rail_m, az)
                Image.fromarray(img[:, :, :3].astype(np.uint8)).save(f"{OUTPUT_DIR}/{name}")
                records.append({
                    "file": name, "frame": frame, "time_sec": t, "pan_offset_deg": pan, "tilt_offset_deg": tilt,
                    "rail_position_m": rail_m, "orbit_azimuth_deg": az, "camera_position_world": cam_eye.tolist(),
                    "human_center_world": center.tolist(), "camera_to_world": as_np(pose).tolist(), "intrinsics_K": K,
                    **view_geometry(cam_eye, gt, center, walk), "joint_depth_gap_m": gaps, "gt_joints_world": gt.tolist(),
                })
            frame_info[frame] = {"time_sec": t, "human_center_world": center.tolist(), "mesh_sync_max_err": sync,
                                 "render_check_median_depth_diff_m": checks}
            write_metadata()
            print(f"Frame {frame}: done ({len(records)} images)")
        print(f"DONE: {len(records)} images -> {OUTPUT_DIR}")
    finally:
        await rep.orchestrator.wait_until_complete_async()
        restore_human()
        rgb.detach()
        depth.detach()
        product.destroy()


sweep_task = asyncio.ensure_future(run())
