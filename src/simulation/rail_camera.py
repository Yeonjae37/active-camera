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
Two ways to move a rig:
  set_pan_tilt()  writes PhysX drive targets (static setup; takes effect on play, real actuator lag).
  RailCamera      kinematic: turns the rig's physics off and writes the carriage / link transforms
                  directly, so rail position, pan and tilt are exact every frame (use this for control).
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

    @classmethod
    def from_prim(cls, stage, path):
        """Rebuild a Rail spawned earlier (e.g. by build_rails.py) from its prim attributes."""
        prim = stage.GetPrimAtPath(path)
        get = lambda n: np.array(prim.GetAttribute(f"rail:{n}").Get(), dtype=float)  # noqa: E731
        return cls(path, get("start"), get("end"), get("inward"))

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
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
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
    x = _xform(stage, path, start, rail.yaw_deg)
    for name, v in (("start", start), ("end", end), ("inward", inward)):
        x.GetPrim().CreateAttribute(f"rail:{name}", Sdf.ValueTypeNames.Double3, custom=True).Set(Gf.Vec3d(*v))
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
    if stage.GetPrimAtPath(rig):  # drop stale overrides from earlier runs (link poses written by PhysX/kinematics)
        stage.RemovePrim(rig)
    _xform(stage, rig, (0, 0, plate_size[2]), yaw_offset_deg)
    stage.GetPrimAtPath(rig).GetReferences().SetReferences([Sdf.Reference(rig_usd)])  # idempotent on re-run
    load_rig_physics(stage, rig)
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
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rig_path), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.FixedJoint) or prim.IsInstanceProxy():
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


def _prims(stage, root):
    """All prims under root, including payload/instance contents (joints can sit inside instances)."""
    return Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())


def load_rig_physics(stage, rig_path):
    """Load payloads under the rig and pick a PhysX variant where a 'physics' variant set is left at none."""
    stage.Load(rig_path)
    for prim in _prims(stage, rig_path):
        for vs_name in prim.GetVariantSets().GetNames():
            vs = prim.GetVariantSets().GetVariantSet(vs_name)
            sel = vs.GetVariantSelection()
            physx = [v for v in vs.GetVariantNames() if "physx" in v.lower()]
            if "physics" in vs_name.lower() and sel.lower() in ("", "none") and physx:
                vs.SetVariantSelection(physx[0])
                print(f"[rail_camera] {prim.GetPath()}: variant {vs_name} '{sel}' -> '{physx[0]}'")


def describe_rig(stage, rig_path) -> str:
    """Joints and variant sets under the rig, for debugging joint lookup."""
    lines = [f"rig {rig_path}:"]
    for prim in _prims(stage, rig_path):
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            lines.append(f"  {prim.GetTypeName():24s} {prim.GetPath()}  body0={j.GetBody0Rel().GetTargets()} "
                         f"body1={j.GetBody1Rel().GetTargets()}")
        for vs_name in prim.GetVariantSets().GetNames():
            vs = prim.GetVariantSets().GetVariantSet(vs_name)
            lines.append(f"  variantSet {prim.GetPath()} {vs_name} = '{vs.GetVariantSelection()}' of {vs.GetVariantNames()}")
    if len(lines) == 1:
        lines.append("  no joints and no variant sets found (rig loaded without physics?)")
    return "\n".join(lines)


def find_pan_tilt_joints(stage, rig_path, pan_name=None, tilt_name=None):
    """(pan_prim, tilt_prim). Exact names if given, else names containing 'pan'/'tilt', else for exactly two
    revolute joints: pan is the one closer to the base (its child link path is shorter)."""
    joints = [p for p in _prims(stage, rig_path) if p.IsA(UsdPhysics.RevoluteJoint)]
    if pan_name or tilt_name:
        pick = {n: next((p for p in joints if p.GetName() == n), None) for n in (pan_name, tilt_name)}
        pan, tilt = pick[pan_name], pick[tilt_name]
    else:
        pan = next((p for p in joints if "pan" in p.GetName().lower()), None)
        tilt = next((p for p in joints if "tilt" in p.GetName().lower()), None)
        if (pan is None or tilt is None) and len(joints) == 2:
            depth = lambda p: len(str((UsdPhysics.Joint(p).GetBody1Rel().GetTargets() or [p.GetPath()])[0]))  # noqa: E731
            pan, tilt = sorted(joints, key=depth)
    if pan is None or tilt is None:
        raise RuntimeError("pan/tilt joints not found\n" + describe_rig(stage, rig_path))
    return pan, tilt


def set_pan_tilt(stage, rig_path, pan_deg, tilt_deg, pan_name=None, tilt_name=None):
    """Joint-space targets [deg] on the angular drives, plus the initial joint state so the rig starts there."""
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    for prim, value in zip(find_pan_tilt_joints(stage, rig_path, pan_name, tilt_name), (pan_deg, tilt_deg)):
        if prim.IsInstanceProxy():
            raise RuntimeError(f"{prim.GetPath()} is inside an instance; make that prim non-instanceable")
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


# ---------------------------------------------------------------- runtime control (kinematic)

def disable_rig_physics(stage, rig_path):
    """Strip physics from the rig for kinematic control: un-instance it (instance proxies cannot be edited),
    remove rigid body / collision / articulation APIs and disable joints. Otherwise its links stay PhysX
    static colliders and every transform write triggers "PxRigidStatic::setGlobalPose: pose is not valid"."""
    apis = [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI,
            UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI]

    def has_physics(prim):
        return any(p.HasAPI(api) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) for api in apis)

    while True:  # un-instance only instances that carry physics; visual instances stay (the renderer keeps them)
        instances = [p for p in Usd.PrimRange(stage.GetPrimAtPath(rig_path)) if p.IsInstance() and has_physics(p)]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rig_path)):
        for api in apis:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)


def sanitize_rig(stage, rig_path) -> list:
    """Drop NaN poses that PhysX wrote over the rig's links (our root-layer overrides only), restoring the
    pose from pan_tilt.usda. Returns the paths that were reset."""
    reset = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(rig_path)):
        if not prim.IsA(UsdGeom.Xformable):
            continue
        m = _np(UsdGeom.Xformable(prim).GetLocalTransformation())
        rest = prim.GetAttribute("railcam:restLocal")
        bad_rest = rest and rest.Get() is not None and not np.isfinite(_np(rest.Get())).all()
        if np.isfinite(m).all() and not bad_rest:
            continue
        for attr in prim.GetAuthoredAttributes():
            name = attr.GetName()
            if name.startswith("xformOp") or name == "railcam:restLocal":
                prim.RemoveProperty(name)
        reset.append(str(prim.GetPath()))
    if reset:
        print(f"[rail_camera] reset NaN poses on {len(reset)} prims: {reset[:5]}")
    return reset


def nan_report(stage, path) -> str:
    """Walk from a prim up to the root and name every prim whose local transform is not finite."""
    lines, prim = [], stage.GetPrimAtPath(path)
    while prim and prim.GetPath() != Sdf.Path.absoluteRootPath:
        if prim.IsA(UsdGeom.Xformable):
            xf = UsdGeom.Xformable(prim)
            m = _np(xf.GetLocalTransformation())
            if not np.isfinite(m).all():
                ops = {op.GetName(): op.Get() for op in xf.GetOrderedXformOps()}
                lines.append(f"  NaN at {prim.GetPath()}  ops={ops}")
        prim = prim.GetParent()
    return "\n".join(lines) or "  no NaN in the transform chain"


def _np(m) -> np.ndarray:
    return np.array(m, dtype=np.float64)


class _KinematicJoint:
    """Revolute joint applied as a transform on its child link: M(q) = F1^-1 R(q) F1 M_rest (row-vector),
    i.e. the child rotates about the joint axis expressed in the child's joint frame."""

    def __init__(self, stage, joint_prim):
        j = UsdPhysics.RevoluteJoint(joint_prim)
        self.link = stage.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
        rest_attr = self.link.GetAttribute("railcam:restLocal")
        if not rest_attr:  # first use: remember the imported rest pose, later runs reuse it
            rest = UsdGeom.Xformable(self.link).GetLocalTransformation()
            rest_attr = self.link.CreateAttribute("railcam:restLocal", Sdf.ValueTypeNames.Matrix4d, custom=True)
            rest_attr.Set(rest)
        self.rest = _np(rest_attr.Get())
        if not np.isfinite(self.rest).all():
            raise RuntimeError(f"{self.link.GetPath()} has an invalid (NaN) pose; re-run build_rails.py to reset the rig")
        q1, p1 = j.GetLocalRot1Attr().Get(), j.GetLocalPos1Attr().Get()
        q1 = Gf.Quatd(q1) if q1 is not None and Gf.Quatd(q1).GetLength() > 1e-6 else Gf.Quatd(1, 0, 0, 0)
        f1 = Gf.Matrix4d().SetRotate(q1.GetNormalized())
        f1.SetTranslateOnly(Gf.Vec3d(p1) if p1 is not None else Gf.Vec3d(0, 0, 0))
        self.f1, self.f1_inv = _np(f1), _np(f1.GetInverse())
        self.axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[j.GetAxisAttr().Get() or "X"]
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        self.limits = (lo if lo is not None and np.isfinite(lo) else -np.inf,
                       hi if hi is not None and np.isfinite(hi) else np.inf)
        self.op = UsdGeom.Xformable(self.link).MakeMatrixXform()
        self.q = 0.0
        print(f"[rail_camera] joint {joint_prim.GetName()}: link {self.link.GetName()} axis {self.axis} "
              f"limits {self.limits} rest finite {np.isfinite(self.rest).all()} frame finite {np.isfinite(self.f1).all()}")

    def local(self, q_deg) -> np.ndarray:
        r = _np(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*self.axis), float(q_deg))))
        return self.f1_inv @ r @ self.f1 @ self.rest

    def set(self, q_deg) -> float:
        if not np.isfinite(q_deg):
            return self.q
        q = float(np.clip((q_deg + 180.0) % 360.0 - 180.0, *self.limits))
        self.q = q
        self.op.Set(Gf.Matrix4d(*self.local(q).flatten().tolist()))
        return q


class RailCamera:
    """Kinematic rail + pan + tilt for one rig. Call set_*/aim_at every frame while the timeline plays;
    camera_to_world() is exact after each call (no drive lag), ready for camera_poses.npy / Calib.toml."""

    def __init__(self, stage, rail: Rail, rig_path, camera_path, pan_joint=None, tilt_joint=None):
        self.stage, self.rail, self.rig, self.camera = stage, rail, rig_path, camera_path
        disable_rig_physics(stage, rig_path)
        sanitize_rig(stage, rig_path)
        pan_prim, tilt_prim = find_pan_tilt_joints(stage, rig_path, pan_joint, tilt_joint)
        self.pan_j, self.tilt_j = _KinematicJoint(stage, pan_prim), _KinematicJoint(stage, tilt_prim)
        self.carriage = UsdGeom.Xformable(stage.GetPrimAtPath(rig_path).GetParent())
        self.s, self.pan, self.tilt = self._read_s(), 0.0, 0.0
        self.set_pan_tilt(0.0, 0.0)
        self.align_camera()
        # fixed pieces of the chain: camera relative to tilt link, tilt link's parent relative to pan link
        tilt_w, pan_w = world_matrix(stage, self.tilt_j.link.GetPath()), world_matrix(stage, self.pan_j.link.GetPath())
        self._cam_in_tilt = world_matrix(stage, camera_path) @ np.linalg.inv(tilt_w)
        self._tiltparent_in_pan = world_matrix(stage, self.tilt_j.link.GetParent().GetPath()) @ np.linalg.inv(pan_w)

    def _axis_world(self, kj):
        """(unit axis, pivot) of a kinematic joint in world coordinates at the current pose."""
        link_w = world_matrix(self.stage, kj.link.GetPath())
        frame_w = kj.f1 @ link_w
        axis = np.asarray(kj.axis, float) @ frame_w[:3, :3]
        return axis / np.linalg.norm(axis), frame_w[3, :3]

    def align_camera(self):
        """Mount the camera upright, looking along the rig's forward direction (perpendicular to the pan and
        tilt axes), whatever orientation the optical frame has in pan_tilt.usda. Forward is the side of the
        tilt axis the camera frame sits on (the lens is in front of the tilt axis), else the carriage +X."""
        up, _ = self._axis_world(self.pan_j)
        up = up if up[2] >= 0 else -up
        tilt_axis, pivot = self._axis_world(self.tilt_j)
        fwd = np.cross(up, tilt_axis)
        fwd /= np.linalg.norm(fwd)
        parent_w = world_matrix(self.stage, self.stage.GetPrimAtPath(self.camera).GetParent().GetPath())
        lens = parent_w[3, :3] - pivot
        lens -= (lens @ up) * up + (lens @ tilt_axis) * tilt_axis
        ref = lens if np.linalg.norm(lens) > 5e-3 else world_matrix(self.stage, str(self.carriage.GetPath()))[0, :3]
        if fwd @ ref < 0:
            fwd = -fwd
        cam_up = up - (up @ fwd) * fwd
        cam_up /= np.linalg.norm(cam_up)
        cam_w = np.eye(4)
        cam_w[0, :3], cam_w[1, :3], cam_w[2, :3], cam_w[3, :3] = np.cross(fwd, cam_up), cam_up, -fwd, parent_w[3, :3]
        local = cam_w @ np.linalg.inv(parent_w)
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.camera))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(*local.flatten().tolist()))

    def _read_s(self) -> float:
        return float(self.carriage.GetOrderedXformOps()[0].Get()[0]) / self.rail.length

    def set_rail_position(self, s):
        self.s = float(np.clip(s, 0.0, 1.0))
        self.carriage.GetOrderedXformOps()[0].Set(Gf.Vec3d(self.s * self.rail.length, 0, 0))

    def set_pan_tilt(self, pan_deg, tilt_deg):
        self.pan, self.tilt = self.pan_j.set(pan_deg), self.tilt_j.set(tilt_deg)

    def _predict(self, pan, tilt) -> np.ndarray:
        """camera_to_world for (pan, tilt) at the current rail position, without touching the stage."""
        pan_parent_w = world_matrix(self.stage, self.pan_j.link.GetParent().GetPath())
        return (self._cam_in_tilt @ self.tilt_j.local(tilt) @ self._tiltparent_in_pan
                @ self.pan_j.local(pan) @ pan_parent_w)

    def aim_at(self, target, iters=4):
        """Pan/tilt that put a world point at the image center (numeric, uses the real joint axes)."""
        if not np.isfinite(target).all():  # keep the last valid pan/tilt
            return self.pan, self.tilt
        target = np.append(np.asarray(target, float), 1.0)

        def err(q):  # target direction in camera coords -> image-plane offsets (x/-z, y/-z)
            p = target @ np.linalg.inv(self._predict(*q))
            return np.array([p[0], p[1]]) / max(-p[2], 1e-6) if p[2] < 0 else np.array([1e3, 1e3])

        q = min(([a, b] for a in (-90, 0, 90, 180) for b in (-45, 0, 45)), key=lambda q: np.linalg.norm(err(q)))
        q = np.array(q, dtype=float) if np.linalg.norm(err([self.pan, self.tilt])) > 1 else np.array([self.pan, self.tilt])
        for _ in range(iters):
            e = err(q)
            jac = np.stack([(err(q + d) - e) / 0.5 for d in (np.array([0.5, 0]), np.array([0, 0.5]))], axis=1)
            q = q - np.linalg.lstsq(jac, e, rcond=None)[0]
        self.set_pan_tilt(*q)
        return self.pan, self.tilt

    def camera_to_world(self) -> np.ndarray:
        return camera_to_world(self.stage, self.camera)

    def intrinsics(self) -> dict:
        return intrinsics(self.stage, self.camera)
