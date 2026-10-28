#!/usr/bin/env python3
"""Two parallel rails with one pan-tilt camera each, person in between (static multi-camera, Step 3).

Rails run along the walking direction (WALK_DIR_DEG, 0 = world +X, the direction of the AMASS walk),
centered on the person, --width apart. Scenes:
  --scene current   (default in the Script Editor) add rails to the stage that is open now
  --scene <file>    open your existing .usd/.usda (human + camera) and add rails to it
  --scene empty     ground plane + lights + a 1.75 m capsule standing in for the person
The person position comes from --person-xy, else from the world position of --person-prim.

--layout facing (default): carriages directly across from each other at the person (rays cross at 180 deg).
--layout offset: both carriages CAM_AHEAD_M ahead of the person (rays cross at ~60-75 deg).
A third static camera looks back at the person from 4.5 m ahead (--no-third to skip).

Run:
  Script Editor (GUI, your scene open): open this file and Run (edit the constants below), or run it with flags:
    import sys, runpy; sys.argv = ["build_rails.py", "--width", "3.0"]
    runpy.run_path("/home/yeonjae/Study/active-camera/scripts/build_rails.py", run_name="__main__")
  Standalone / ssh, from the repo root with the Isaac Sim env active:
    python scripts/build_rails.py --headless --scene ~/scenes/my_scene.usda --width 3.0 \
        --render output/step3/w3.0 --save output/step3/w3.0/stage.usda
    for w in 2.0 2.5 3.0 3.5; do python scripts/build_rails.py --headless --scene ~/scenes/my_scene.usda \
        --width $w --render output/step3/w$w; done
  --save writes a NEW file; your original scene file is never overwritten.
Paths: env ACTIVE_CAMERA_ROOT (repo with src/) and PAN_TILT_USD (rig .usda) override the defaults.
"""

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

IN_KIT = "omni.kit.app" in sys.modules  # True inside the Script Editor: never start a second SimulationApp there

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="current" if IN_KIT else "empty", help="current | empty | path to .usd")
parser.add_argument("--person-prim", default="/World/WalkingHuman")
parser.add_argument("--person-xy", default=None, help="'x,y' [m], overrides --person-prim")
parser.add_argument("--width", type=float, default=None, help="rail-to-rail distance [m]")
parser.add_argument("--rail-length", type=float, default=None, help="[m]")
parser.add_argument("--rail-height", type=float, default=None, help="top of the beam [m]")
parser.add_argument("--aim-z", type=float, default=None, help="height the cameras aim at on the person [m]")
parser.add_argument("--layout", choices=("facing", "offset"), default="facing")
parser.add_argument("--no-third", action="store_true", help="skip the static third camera")
parser.add_argument("--render", default=None, help="write one PNG per camera to this folder (works headless)")
parser.add_argument("--save", default=None, help="write the stage with rails to this NEW .usda")
parser.add_argument("--headless", action="store_true")
# Script Editor: Kit's own argv is ignored unless a runner set sys.argv = ["build_rails.py", ...flags]
_kit_argv = sys.argv[1:] if sys.argv and Path(sys.argv[0]).name == "build_rails.py" else []
args = parser.parse_args(_kit_argv if IN_KIT else None)
if not IN_KIT:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

REPO_ROOT = Path(os.environ.get("ACTIVE_CAMERA_ROOT", Path.home() / "Study/active-camera"))  # if not run from inside the repo
_here = Path(__file__).resolve().parents[1] if "__file__" in globals() else None
ROOT = _here if _here is not None and (_here / "src/simulation/rail_camera.py").exists() else REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))
from simulation import rail_camera as rc  # noqa: E402

rc = importlib.reload(rc)  # Script Editor keeps modules between runs

# ---------------------------------------------------------------- config (CLI flags override)
RIG_USD = os.environ.get("PAN_TILT_USD", str(ROOT / "assets/pantilt_urdf/pan_tilt/pan_tilt.usda"))
WIDTH = 3.0                      # rail-to-rail [m]
RAIL_LENGTH = 8.0                # [m], centered on the person
RAIL_HEIGHT = 2.1                # top of the beam [m]
PERSON_AIM_Z = 1.1               # aim point height [m] (pelvis/chest)
PERSON_XY = None                 # e.g. (2.6, -1.0) = middle of the AMASS walk; None -> read --person-prim
WALK_DIR_DEG = 0.0               # rail direction in world xy, 0 = +X
CAM_AHEAD_M = 1.8                # --layout offset: both carriages this far ahead of the person
RIG_YAW_OFFSET_DEG = 0.0         # set to 90/-90/180 if the rig's pan=0 direction is not its local +X


def person_xy(stage):
    if args.person_xy:
        return tuple(float(v) for v in args.person_xy.split(","))
    if PERSON_XY is not None:
        return PERSON_XY
    if args.scene == "empty":
        return (0.0, 0.0)
    prim = stage.GetPrimAtPath(args.person_prim)
    if not prim:
        raise RuntimeError(f"{args.person_prim} not in the stage: pass --person-prim or --person-xy (stage unchanged)")
    t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return (t[0], t[1])


def build_empty(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if not stage.GetPrimAtPath("/World/Ground"):
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.47, 0.5)])
        ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
        ground.AddScaleOp().Set(Gf.Vec3f(30, 30, 0.01))
        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(800.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(2500.0)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(35, 0, 30))
    if not stage.GetPrimAtPath("/World/PersonProxy"):
        proxy = UsdGeom.Capsule.Define(stage, "/World/PersonProxy")  # 1.75 m stand-in to check framing
        proxy.CreateRadiusAttr(0.2)
        proxy.CreateHeightAttr(1.35)
        proxy.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.3)])
        proxy.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.875))


def build(stage):
    if not Path(RIG_USD).exists():
        raise FileNotFoundError(f"RIG_USD not found: {RIG_USD} (set PAN_TILT_USD; stage unchanged)")
    center = np.asarray(person_xy(stage), dtype=float)
    width = args.width or WIDTH
    height = args.rail_height or RAIL_HEIGHT
    aim_z = args.aim_z or PERSON_AIM_Z
    walk = np.array([math.cos(math.radians(WALK_DIR_DEG)), math.sin(math.radians(WALK_DIR_DEG))])
    half = 0.5 * (args.rail_length or RAIL_LENGTH)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Rails")
    if args.scene == "empty":
        build_empty(stage)
    if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    person = np.array([*center, aim_z])
    cam_xy = center + walk * (CAM_AHEAD_M if args.layout == "offset" else 0.0)
    cams = {}
    for i, (w0, w1, inward) in enumerate(rc.corridor_walls(center - walk * half, center + walk * half, width), 1):
        rail = rc.spawn_rail(stage, f"/World/Rails/rail_{i}", w0, w1, inward, height=height, wall_offset=0.0)
        rig = rc.place_rig(stage, rail, "carriage", RIG_USD, s=rail.s_at(cam_xy), yaw_offset_deg=RIG_YAW_OFFSET_DEG)
        cam = rc.attach_camera(stage, rig)
        pan, tilt = rc.aim_pan_tilt(stage, rig, cam, person)
        rc.set_pan_tilt(stage, rig, pan, tilt)
        cams[f"cam{i:02d}"] = (rig, cam)
        print(f"[rails] cam{i:02d}: s={rail.s_at(cam_xy):.3f} pan={pan:.1f} tilt={tilt:.1f}  {cam}")

    if not args.no_third:
        eye = np.array([*(center + walk * 4.5), 1.6])
        cams["cam03"] = (None, rc.add_static_camera(stage, "/World/Rails/cam03", eye, person))

    # crossing angle of the two rail cameras at the person, from the authored (pan = tilt = 0) positions
    a, b = (rc.camera_to_world(stage, cams[k][1])[3, :3] - person for k in ("cam01", "cam02"))
    cos = np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0)
    print(f"[rails] person at {center.round(3)}, width={width:.2f} m, rail height={height:.2f} m, "
          f"camera-person {np.linalg.norm(a):.2f} m, 3D ray crossing angle {math.degrees(math.acos(cos)):.1f} deg")
    return cams


def render_previews(cams, out_dir, size=(1920, 1080)):
    """One RGB frame per camera through Replicator (no viewport needed), to check framing over ssh."""
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    annots = {}
    for name, (_, cam) in cams.items():
        annots[name] = rep.AnnotatorRegistry.get_annotator("rgb")
        annots[name].attach([rep.create.render_product(cam, size)])
    for _ in range(3):  # first frames can be black while RTX warms up
        try:
            rep.orchestrator.step(rt_subframes=16)
        except TypeError:
            rep.orchestrator.step()
    for name, ann in annots.items():
        Image.fromarray(np.asarray(ann.get_data())[..., :3]).save(out_dir / f"{name}.png")
        print(f"[rails] wrote {out_dir / f'{name}.png'}")


def wait_loaded():
    for _ in range(3000):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            break


if not IN_KIT and args.scene not in ("current", "empty"):
    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    if args.save and Path(args.save).expanduser().resolve() == scene:
        raise ValueError("--save must differ from --scene so the original scene is kept")
    omni.usd.get_context().open_stage(str(scene))
    wait_loaded()

stage = omni.usd.get_context().get_stage()
cams = build(stage)
for name, (_, cam) in cams.items():
    print(f"[rails] {name} intrinsics: {rc.intrinsics(stage, cam)}")

if IN_KIT:
    print("[rails] done. Press Play so the pan/tilt drives move to their targets. Save As to keep it.")
else:
    wait_loaded()
    omni.timeline.get_timeline_interface().play()  # drives move the rigs to their pan/tilt targets
    for _ in range(120):
        app.update()
    if args.render:
        render_previews(cams, Path(args.render))
    if args.save:
        out = Path(args.save).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out))
        print(f"[rails] saved {out}")
    while not args.headless and app.is_running():
        app.update()
    app.close()
