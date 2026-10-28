"""Wall-mounted linear rails carrying pan-tilt RGB cameras, for Isaac Sim (static placement, pure USD).

Layout:
  <rail>                Xform at the rail start, rotateZ = rail direction (local +X along the rail)
    beam                visual box, length L
    <carriage>          Xform at local x = s * L, rotateZ so its +X points into the corridor
      plate             visual box the rig sits on
      pan_tilt          reference to pan_tilt.usda (rig forward = carriage +X after yaw_offset_deg)
        .../camera_color_optical_frame/FemtoBolt_RGB   UsdGeom.Camera

Conventions are the recorder's: Z-up meters, camera_to_world() is the USD row-vector 4x4 (same as
camera_poses.npy), intrinsics() returns the metadata.json "intrinsics" dict (ideal pinhole).
Pan/tilt are joint-space degrees written to the rig's angular drives; they take effect once the
timeline plays. Move carriages before pressing play (the fixed base joint is re-anchored on each move).
"""

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# Femto Bolt RGB 1920x1080, 80 deg HFOV: identical to metadata.json of the single-camera recordings
FEMTO_BOLT = dict(width=1920, height=1080, focal_length=11.917535781860352,
                  horizontal_aperture=20.0, vertical_aperture=11.25)


@dataclass
class Rail:
    path: str
    start: np.ndarray    # world xyz of the rail axis start (top of the beam)
    end: np.ndarray
    inward: np.ndarray   # unit xy vector from the wall into the corridor

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def yaw_deg(self) -> float:
        d = self.end - self.start
        return math.degrees(math.atan2(d[1], d[0]))

    @property
    def inward_local_deg(self) -> float:
        """Carriage yaw (in the rail frame) that faces the corridor: +90 if inward is the rail's left."""
        d = (self.end - self.start)[:2]
        return 90.0 if d[0] * self.inward[1] - d[1] * self.inward[0] > 0 else -90.0

    def point(self, s: float) -> np.ndarray:
        return self.start + float(np.clip(s, 0.0, 1.0)) * (self.end - self.start)

    def s_at(self, xy) -> float:
        """Rail parameter of the point on the rail closest to a world xy."""
        d = (self.end - self.start)[:2]
        return float(np.clip(np.dot(np.asarray(xy, float)[:2] - self.start[:2], d) / np.dot(d, d), 0.0, 1.0))


# ---------------------------------------------------------------- helpers

def _xform(stage, path, translate=(0, 0, 0), yaw_deg=0.0, scale=None):
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    x.AddRotateZOp().Set(float(yaw_deg))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    return x


def _box(stage, path, size, center=(0, 0, 0), color=(0.6, 0.6, 0.62)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube.ClearXformOpOrder()
    cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    return cube


def _find(stage, root, match):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if match(prim):
            return prim
    return None


def world_matrix(stage, path) -> np.ndarray:
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=np.float64)


def look_at_matrix(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """USD row-vector camera-to-world: camera looks along -Z, +Y up."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, -f, eye
    return m


# ---------------------------------------------------------------- rails

def corridor_walls(center_start, center_end, width):
    """Corridor centerline (xy) + width -> [(wall_start, wall_end, inward)] for the left and right walls."""
    a, b = np.asarray(center_start, float)[:2], np.asarray(center_end, float)[:2]
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])  # left of the walking direction
    h = 0.5 * width
    return [(a + n * h, b + n * h, -n), (a - n * h, b - n * h, n)]


def spawn_rail(stage, path, wall_start, wall_end, inward, height=2.0, wall_offset=0.10,
               beam_size=(0.06, 0.04)) -> Rail:
    """Visual rail along a wall. wall_* are xy points on the wall surface; height = top of the beam [m]."""
    inward = np.append(np.asarray(inward, float)[:2], 0.0)
    inward /= np.linalg.norm(inward)
    lift = np.array([0.0, 0.0, height])
    start = np.append(np.asarray(wall_start, float)[:2], 0.0) + lift + inward * wall_offset
    end = np.append(np.asarray(wall_end, float)[:2], 0.0) + lift + inward * wall_offset
    rail = Rail(path, start, end, inward)

    w, h = beam_size
    _xform(stage, path, start, rail.yaw_deg)
    _box(stage, f"{path}/beam", (rail.length, w, h), (0.5 * rail.length, 0, -0.5 * h))
    # PhysX prismatic joint along local +X can be added here later for a driven carriage.
    return rail


# ---------------------------------------------------------------- rig + camera

def place_rig(stage, rail: Rail, name, rig_usd, s=0.5, yaw_offset_deg=0.0, plate_size=(0.14, 0.14, 0.02)) -> str:
    """Carriage at rail parameter s with the pan-tilt rig referenced on top. Returns the rig prim path."""
    carriage = f"{rail.path}/{name}"
    _xform(stage, carriage, (s * rail.length, 0, 0), rail.inward_local_deg)
    _box(stage, f"{carriage}/plate", plate_size, (0, 0, 0.5 * plate_size[2]), color=(0.25, 0.25, 0.28))

    rig = f"{carriage}/pan_tilt"
    _xform(stage, rig, (0, 0, plate_size[2]), yaw_offset_deg)
    stage.GetPrimAtPath(rig).GetReferences().SetReferences([Sdf.Reference(rig_usd)])  # idempotent on re-run
    reanchor_fixed_base(stage, rig)
    return rig


def set_rail_position(stage, rail: Rail, rig_path, s):
    """Slide the carriage that holds rig_path to rail parameter s in [0, 1] (call before play)."""
    carriage = stage.GetPrimAtPath(rig_path).GetParent()
    op = UsdGeom.Xformable(carriage).GetOrderedXformOps()[0]  # translate
    op.Set(Gf.Vec3d(float(np.clip(s, 0, 1)) * rail.length, 0, 0))
    reanchor_fixed_base(stage, rig_path)


def reanchor_fixed_base(stage, rig_path):
    """An imported fixed-base rig has a world->base_link FixedJoint whose world frame was baked at import;
    move it to base_link's current pose so PhysX does not pull the rig back to the origin on play."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rig_path)):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body1 = joint.GetBody1Rel().GetTargets()
        if joint.GetBody0Rel().GetTargets() or not body1:
            continue
        m = UsdGeom.Xformable(stage.GetPrimAtPath(body1[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q = m.RemoveScaleShear().ExtractRotationQuat()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(m.ExtractTranslation()))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))


def _set_camera_intrinsics(cam: UsdGeom.Camera, intr):
    cam.CreateFocalLengthAttr(float(intr["focal_length"]))
    cam.CreateHorizontalApertureAttr(float(intr["horizontal_aperture"]))
    cam.CreateVerticalApertureAttr(float(intr["vertical_aperture"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))


def attach_camera(stage, rig_path, name="FemtoBolt_RGB", frame="camera_color_optical_frame", intr=FEMTO_BOLT) -> str:
    """RGB camera under the rig's optical frame (ROS: +Z forward, +Y down -> USD camera needs rotateX 180).
    Falls back to the tilt link (camera then looks along the link's -Z) if the frame is missing."""
    frame_prim = _find(stage, rig_path, lambda p: p.GetName() == frame)
    optical = frame_prim is not None
    if not optical:
        frame_prim = _find(stage, rig_path, lambda p: "tilt" in p.GetName().lower() and not p.IsA(UsdPhysics.Joint))
    path = f"{frame_prim.GetPath()}/{name}"

    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsA(UsdGeom.Camera):  # already authored inside pan_tilt.usda
        _set_camera_intrinsics(UsdGeom.Camera(existing), intr)
        return path
    cam = UsdGeom.Camera.Define(stage, path)
    _set_camera_intrinsics(cam, intr)
    cam.ClearXformOpOrder()
    if optical:
        cam.AddRotateXOp().Set(180.0)
    return path


def add_static_camera(stage, path, eye, target, intr=FEMTO_BOLT) -> str:
    cam = UsdGeom.Camera.Define(stage, path)
    _set_camera_intrinsics(cam, intr)
    cam.ClearXformOpOrder()
    cam.AddTransformOp().Set(Gf.Matrix4d(*look_at_matrix(eye, target).flatten().tolist()))
    return path


def _joint(stage, rig_path, key):
    prim = _find(stage, rig_path, lambda p: p.IsA(UsdPhysics.RevoluteJoint) and key in p.GetName().lower())
    if prim is None:
        raise RuntimeError(f"no revolute joint containing '{key}' under {rig_path}")
    return prim


def set_pan_tilt(stage, rig_path, pan_deg, tilt_deg, pan_key="pan", tilt_key="tilt"):
    """Joint-space targets [deg] on the angular drives, plus the initial joint state so the rig starts there."""
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    for key, value in ((pan_key, pan_deg), (tilt_key, tilt_deg)):
        prim = _joint(stage, rig_path, key)
        UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTargetPositionAttr().Set(float(value))
        if PhysxSchema is not None:
            PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr().Set(float(value))


def aim_pan_tilt(stage, rig_path, camera_path, target, pan_sign=1.0, tilt_sign=1.0):
    """Approximate (pan, tilt) [deg] that point the camera at a world point, ignoring link offsets.
    Assumes pan = +rotation about the carriage +Z and tilt > 0 looks down (tilt_target 25 looked down
    in the recordings). Flip pan_sign / tilt_sign if the rig's joint axes say otherwise."""
    carriage = str(stage.GetPrimAtPath(rig_path).GetParent().GetPath())
    to_local = np.linalg.inv(world_matrix(stage, carriage))
    p = np.append(np.asarray(target, float), 1.0) @ to_local
    cam_z = (np.array([0, 0, 0, 1.0]) @ world_matrix(stage, camera_path) @ to_local)[2]  # at pan = tilt = 0
    pan = math.degrees(math.atan2(p[1], p[0]))
    tilt = math.degrees(math.atan2(cam_z - p[2], math.hypot(p[0], p[1])))
    return pan_sign * pan, tilt_sign * tilt


# ---------------------------------------------------------------- export

def camera_to_world(stage, camera_path) -> np.ndarray:
    """4x4 USD row-vector camera-to-world, same convention as camera_poses.npy. With Fabric/physics
    running, read it the way the recorder already does if USD is not being written back."""
    return world_matrix(stage, camera_path)


def intrinsics(stage, camera_path, width=FEMTO_BOLT["width"], height=FEMTO_BOLT["height"]) -> dict:
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    f = cam.GetFocalLengthAttr().Get()
    ha, va = cam.GetHorizontalApertureAttr().Get(), cam.GetVerticalApertureAttr().Get()
    return dict(width=width, height=height, fx=f / ha * width, fy=f / va * height,
                cx=width / 2.0, cy=height / 2.0, focal_length=f, horizontal_aperture=ha,
                vertical_aperture=va, model="ideal_pinhole")
