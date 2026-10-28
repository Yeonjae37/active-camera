#!/usr/bin/env python3
"""Hospital corridor with a rail on each side wall and one pan-tilt camera per rail (static, Step 3).

--layout facing (default): the two carriages sit directly across from each other at PERSON_XY, so the
cameras face each other with the person in between (ray crossing angle 180 deg).
--layout offset: both carriages CAM_AHEAD_M ahead of PERSON_XY, rays cross at 2 * atan(half_width /
CAM_AHEAD_M) (~60-75 deg), better triangulation geometry. An optional third static camera looks down the corridor.

Two ways to run:
  Script Editor (Isaac Sim GUI already open): Window > Script Editor, open/paste this file, Run.
      Adds rails and cameras to the CURRENT stage; references hospital.usd under /World/hospital only if
      it is not there yet. Set REPO_ROOT below if your repo is not ~/Study/active-camera.
  Standalone (from the repo root):
      ~/isaacsim/python.sh scripts/build_hospital_rails.py            # opens the GUI, plays, stays open
      ~/isaacsim/python.sh scripts/build_hospital_rails.py --save output/step3/hospital_rails.usd --headless

CORRIDOR_* / PERSON_XY come from scripts/measure_corridor.py.
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

IN_KIT = "omni.kit.app" in sys.modules  # True inside the Script Editor: never start a second SimulationApp there

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--save", default=None, help="write the composed stage to this .usd")
parser.add_argument("--layout", choices=("facing", "offset"), default="facing")
parser.add_argument("--no-third", action="store_true", help="skip the static end-of-corridor camera")
args = parser.parse_args([] if IN_KIT else None)  # Script Editor: edit the defaults above instead
if not IN_KIT:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom, UsdPhysics  # noqa: E402

REPO_ROOT = Path.home() / "Study/active-camera"  # used when this file is not run from inside the repo
_here = Path(__file__).resolve().parents[1] if "__file__" in globals() else None
ROOT = _here if _here is not None and (_here / "src/simulation/rail_camera.py").exists() else REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))
from simulation import rail_camera as rc  # noqa: E402

rc = importlib.reload(rc)  # Script Editor keeps modules between runs

# ---------------------------------------------------------------- scene config (TODO: measure in the GUI)
HOSPITAL_REL = "/Isaac/Environments/Hospital/hospital.usd"
HOSPITAL_FALLBACK = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2" + HOSPITAL_REL
RIG_USD = str(ROOT / "assets/pantilt_urdf/pan_tilt/pan_tilt.usda")
CORRIDOR_START = (-13.041, 10.555)  # centerline xy (measure_corridor.py), walking direction = start -> end
CORRIDOR_END = (3.158, 10.555)
CORRIDOR_WIDTH = 3.497             # wall-to-wall [m]
PERSON_XY = (-4.941, 10.555)      # where the motion happens (on the centerline)
PERSON_AIM_Z = 1.1               # aim point height [m] (pelvis/chest)
RAIL_HEIGHT = 2.1                # top of the beam [m], above the handrails and door frames
CAM_AHEAD_M = 1.8                # --layout offset: both carriages this far ahead of the person
RIG_YAW_OFFSET_DEG = 0.0         # set to 90/-90/180 if the rig's pan=0 direction is not its local +X
HUMAN_PRIM = "/World/WalkingHuman"


def assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path
        return get_assets_root_path()
    except Exception:
        return None


def build(stage):
    if not Path(RIG_USD).exists():
        raise FileNotFoundError(f"RIG_USD not found: {RIG_USD} (fix RIG_USD / REPO_ROOT; the stage was not changed)")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = assets_root()
    UsdGeom.Xform.Define(stage, "/World")
    if not stage.GetPrimAtPath("/World/hospital"):
        stage.DefinePrim("/World/hospital", "Xform").GetReferences().AddReference(
            root + HOSPITAL_REL if root else HOSPITAL_FALLBACK)
    if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    person = np.array([*PERSON_XY, PERSON_AIM_Z])
    walk = np.subtract(CORRIDOR_END, CORRIDOR_START)
    walk = walk / np.linalg.norm(walk)
    cam_xy = np.asarray(PERSON_XY) + walk * (CAM_AHEAD_M if args.layout == "offset" else 0.0)

    cams = {}
    for i, (w0, w1, inward) in enumerate(rc.corridor_walls(CORRIDOR_START, CORRIDOR_END, CORRIDOR_WIDTH), 1):
        rail = rc.spawn_rail(stage, f"/World/Rails/rail_{i}", w0, w1, inward, height=RAIL_HEIGHT)
        rig = rc.place_rig(stage, rail, "carriage", RIG_USD, s=rail.s_at(cam_xy), yaw_offset_deg=RIG_YAW_OFFSET_DEG)
        cam = rc.attach_camera(stage, rig)
        pan, tilt = rc.aim_pan_tilt(stage, rig, cam, person)
        rc.set_pan_tilt(stage, rig, pan, tilt)
        cams[f"cam{i:02d}"] = (rig, cam)
        print(f"[rails] cam{i:02d}: s={rail.s_at(cam_xy):.3f} pan={pan:.1f} tilt={tilt:.1f}  {cam}")

    if not args.no_third:  # static, at the far end of the corridor, looking back at the person
        eye = np.array([*(np.asarray(PERSON_XY) + walk * 4.5), 1.6])
        cams["cam03"] = (None, rc.add_static_camera(stage, "/World/Cameras/cam03", eye, person))

    # crossing angle of the two rail cameras at the person, from the authored (pan = tilt = 0) positions
    p = [rc.camera_to_world(stage, c)[3, :3] for k, (_, c) in cams.items() if k != "cam03"]
    a, b = p[0] - person, p[1] - person
    print(f"[rails] ray crossing angle: {math.degrees(math.acos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))):.1f} deg")

    human = stage.GetPrimAtPath(HUMAN_PRIM)
    if human:  # the walking human's root, if it is already in this stage
        UsdGeom.XformCommonAPI(human).SetTranslate((*PERSON_XY, 0.0))
    return cams


stage = omni.usd.get_context().get_stage()
cams = build(stage)
for name, (_, cam) in cams.items():
    print(f"[rails] {name} intrinsics: {rc.intrinsics(stage, cam)}")

if IN_KIT:
    print("[rails] done. Press Play so the pan/tilt drives move to their targets.")
else:
    for _ in range(10):
        app.update()
    omni.timeline.get_timeline_interface().play()  # drives move the rigs to their pan/tilt targets
    for _ in range(120):
        app.update()
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(args.save)
        print(f"[rails] saved {args.save}")
    while not args.headless and app.is_running():
        app.update()
    app.close()
