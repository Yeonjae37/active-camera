"""Drive the rail cameras built by build_rails.py (Script Editor, kinematic): cam01 slides back and forth along
its rail, cam02 stays put, both keep the human's pelvis at the image center.

MODE = "record" (default): steps the animation frame by frame itself (no Play needed), renders every camera
    with Replicator after the pose is set, and writes one folder per camera in the recorder's format:
        OUT_ROOT/<timestamp>/cam01/000000.png ... camera_poses.npy frames.json metadata.json timestamps.csv
        (+ joints_gt.npy, the skeleton joints in world coordinates, when the human is a UsdSkel)
    Image and pose come from the same step, so the folders feed pose2sim_calib.py directly.
MODE = "live": controls the cameras while you press Play (viewport only) and saves the poses when you stop.

Run in the Script Editor with the scene (with rails) open:
    import runpy, os; runpy.run_path(os.path.expanduser("~/Study/active-camera/scripts/run_rail_demo.py"), run_name="__main__")
Re-running replaces the previous controller.
"""

import asyncio
import builtins
import csv
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdSkel

REPO_ROOT = Path(os.environ.get("ACTIVE_CAMERA_ROOT", Path.home() / "Study/active-camera"))


def _load_rail_camera(path):
    """Fresh exec of rail_camera.py on every run: the Script Editor's Python keeps old modules and .pyc."""
    mod = types.ModuleType("simulation.rail_camera")
    mod.__file__ = str(path)
    sys.modules["simulation.rail_camera"] = mod
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


rc = _load_rail_camera(REPO_ROOT / "src/simulation/rail_camera.py")
print(f"[rail_demo] rail_camera from {rc.__file__} (RailCamera: {hasattr(rc, 'RailCamera')})")

MODE = "record"           # "record" | "live" | "off"
HUMAN_PRIM = "/World/WalkingHuman"
RAILS = "/World/Rails"
SWEEP_PERIOD_S = 6.0      # cam01: one back-and-forth along the rail
SWEEP_RANGE = (0.05, 0.95)  # cam01 rail parameter range; None = cam01 stays where build_rails put it
FOLLOW = True             # aim at the pelvis every frame; False = keep the pan/tilt (static, for Pose2Sim)
RECORD_FPS = 10.0         # frames saved per second of animation (animation itself is usually 30 fps)
RECORD_SECONDS = None     # None = the stage's whole animation range
RESOLUTION = (1920, 1080)  # matches the Femto Bolt intrinsics in metadata.json
RT_SUBFRAMES = 4          # extra RTX passes per frame (less noise/ghosting, slower)
OUT_ROOT = REPO_ROOT / "output/step3/rail_demo"


def skeleton_query(stage):
    root = stage.GetPrimAtPath(HUMAN_PRIM)
    skel_prim = next((p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton)), None) if root else None
    if skel_prim is None:
        return None
    cache = UsdSkel.Cache()
    skel_root = UsdSkel.Root.Find(skel_prim)
    if skel_root:
        cache.Populate(skel_root, Usd.PrimDefaultPredicate)
    return cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim)) or None


def joints_world(query, time_code):
    """(J, 3) skeleton joint positions in world coordinates, or None."""
    if query is None:
        return None
    xf = query.ComputeJointWorldTransforms(UsdGeom.XformCache(time_code))
    return np.array([m.ExtractTranslation() for m in xf]) if xf else None


def pelvis_world(stage, query, time_code):
    """SMPL root joint (pelvis); falls back to the human's bbox center, then its root + 1 m."""
    joints = joints_world(query, time_code)
    if joints is not None:
        return joints[0]
    root = stage.GetPrimAtPath(HUMAN_PRIM)
    box = UsdGeom.BBoxCache(time_code, [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedRange()
    if not box.IsEmpty():
        return (np.array(box.GetMin()) + np.array(box.GetMax())) / 2
    t = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time_code).ExtractTranslation()
    return np.array([t[0], t[1], 1.0])


def make_cameras(stage):
    """[(name, camera_path, RailCamera or None)] for every rail camera plus static cameras under RAILS."""
    cams = []
    for prim in stage.GetPrimAtPath(RAILS).GetChildren():
        name = prim.GetName()
        if name.startswith("rail_"):  # rail_1 -> cam01
            rig = f"{prim.GetPath()}/carriage/pan_tilt"
            cam = rc.attach_camera(stage, rig)  # returns the existing camera
            ctl = rc.RailCamera(stage, rc.Rail.from_prim(stage, str(prim.GetPath())), rig, cam)
            cams.append((f"cam{int(name.split('_')[1]):02d}", cam, ctl))
        elif prim.IsA(UsdGeom.Camera):  # static cameras keep their prim name (cam03)
            cams.append((name, str(prim.GetPath()), None))
    return sorted(cams, key=lambda c: (c[2] is None, c[0]))  # rail cameras first: control() moves cams[0]


def control(cams, target, t):
    """The demo policy: cam01 sweeps along its rail, every rail camera aims at the target."""
    rails = [ctl for _, _, ctl in cams if ctl is not None]
    if not getattr(control, "printed", False):
        print(f"[rail_demo] first aim target (pelvis) = {np.round(target, 3)} from {HUMAN_PRIM}")
        control.printed = True
    if not np.isfinite(target).all():
        print(f"[rail_demo] WARNING invalid target at t={t:.2f}; keeping the previous pan/tilt")
    if SWEEP_RANGE:
        lo, hi = SWEEP_RANGE
        rails[0].set_rail_position(lo + (hi - lo) * 0.5 * (1 - math.cos(2 * math.pi * t / SWEEP_PERIOD_S)))
    if FOLLOW or SWEEP_RANGE or not getattr(control, "aimed", False):  # static: aim once at the first frame
        for ctl in rails:
            ctl.aim_at(target)
        control.aimed = True


async def record(stage, cams):
    import omni.replicator.core as rep
    from PIL import Image

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    tcps = stage.GetTimeCodesPerSecond()
    t0 = stage.GetStartTimeCode() / tcps
    t1 = t0 + RECORD_SECONDS if RECORD_SECONDS else stage.GetEndTimeCode() / tcps
    times = np.arange(t0, t1 + 1e-9, 1.0 / RECORD_FPS)
    query = skeleton_query(stage)
    out = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")

    products, annots, logs = [], {}, {}
    for name, path, ctl in cams:
        rp = rep.create.render_product(path, RESOLUTION)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        products.append(rp)
        annots[name] = ann
        logs[name] = dict(poses=[], frames=[], joints=[])
        (out / name).mkdir(parents=True, exist_ok=True)
    print(f"[rail_demo] recording {len(times)} frames ({t0:.2f}-{t1:.2f} s) x {len(cams)} cameras to {out}")
    if len(times) < 2:
        print("[rail_demo] WARNING: the stage has no animation range (start = end time code); set RECORD_SECONDS")

    async def render():
        try:
            await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=True)
        except TypeError:  # older Replicator signature
            await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)

    for name, path, ctl in cams:  # state before anything moves
        print(f"[rail_demo] {name} start pose finite={np.isfinite(rc.camera_to_world(stage, path)).all()}")
        if not np.isfinite(rc.camera_to_world(stage, path)).all():
            print(rc.nan_report(stage, path))
    await render()  # warm-up: the first RTX frame is often black
    for k, t in enumerate(times):
        tc = Usd.TimeCode(t * tcps)
        timeline.set_current_time(t)
        control(cams, pelvis_world(stage, query, tc), t)
        await render()
        bad = [name for name, path, _ in cams if not np.isfinite(rc.camera_to_world(stage, path)).all()]
        if bad:
            report = "\n".join(f"{name}:\n{rc.nan_report(stage, path)}" for name, path, _ in cams if name in bad)
            raise RuntimeError(f"NaN camera pose at t={t:.2f}\n{report}")
        joints = joints_world(query, tc)
        for name, path, ctl in cams:
            image = f"{k:06d}.png"
            Image.fromarray(np.asarray(annots[name].get_data())[..., :3]).save(out / name / image)
            log = logs[name]
            log["poses"].append(rc.camera_to_world(stage, path))
            log["frames"].append(dict(
                frame=k, image=image, target_time_sec=float(t), actual_time_sec=float(timeline.get_current_time()),
                pan_target_deg=ctl.pan if ctl else None, tilt_target_deg=ctl.tilt if ctl else None,
                pan_actual_deg=ctl.pan if ctl else None, tilt_actual_deg=ctl.tilt if ctl else None,
                rail_s=ctl.s if ctl else None, joint_sensor_time_sec=float(t)))
            if joints is not None:
                log["joints"].append(joints)
        if k % 10 == 0 or k == len(times) - 1:
            pelvis = pelvis_world(stage, query, tc)

            def px(path):  # where the pelvis lands in the image (should be ~ the center 960, 540)
                p = np.append(pelvis, 1.0) @ np.linalg.inv(rc.camera_to_world(stage, path))
                f = RESOLUTION[0] * 11.917535781860352 / 20.0
                return (round(RESOLUTION[0] / 2 + f * p[0] / -p[2]), round(RESOLUTION[1] / 2 - f * p[1] / -p[2]))

            state = "  ".join(f"{name}: s={ctl.s:.2f} xyz={np.round(rc.camera_to_world(stage, path)[3, :3], 2)} "
                              f"pan={ctl.pan:.0f} tilt={ctl.tilt:.0f} pelvis_px={px(path)}"
                              for name, path, ctl in cams if ctl)
            print(f"[rail_demo] frame {k}/{len(times)} t={t:.2f}  {state}")

    for name, path, ctl in cams:
        d, log = out / name, logs[name]
        np.save(d / "camera_poses.npy", np.asarray(log["poses"]))
        if log["joints"]:
            np.save(d / "joints_gt.npy", np.asarray(log["joints"]))
        (d / "frames.json").write_text(json.dumps(log["frames"], indent=2))
        with open(d / "timestamps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log["frames"][0]))
            w.writeheader()
            w.writerows(log["frames"])
        intr = rc.intrinsics(stage, path, *RESOLUTION)
        (d / "metadata.json").write_text(json.dumps(dict(
            mode="rail_kinematic" if ctl else "static", camera_path=path, human_path=HUMAN_PRIM,
            captured_frames=len(times), record_fps=RECORD_FPS, stage_time_codes_per_second=tcps,
            meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), up_axis=UsdGeom.GetStageUpAxis(stage),
            intrinsics=intr, camera_matrix_convention="USD row-vector, camera local to world",
            joint_gt_reference="UsdSkel joints of the human, world coordinates, same time as the image",
            notes="Kinematic rail camera: pose set, then rendered with Replicator in the same step (no lag)."), indent=2))
    for ann in annots.values():
        ann.detach()
    for rp in products:
        rp.destroy()
    print(f"[rail_demo] done: {out}")


def start_live(stage, cams):
    timeline = omni.timeline.get_timeline_interface()
    query = skeleton_query(stage)
    log = {"time": [], **{name: [] for name, _, _ in cams}}
    out = OUT_ROOT / "live_poses.npz"

    def on_update(_event):
        if not timeline.is_playing():
            if log["time"]:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, **{k: np.asarray(v) for k, v in log.items()})
                print(f"[rail_demo] saved {len(log['time'])} poses to {out}")
                for v in log.values():
                    v.clear()
            return
        t = timeline.get_current_time()
        control(cams, pelvis_world(stage, query, Usd.TimeCode(t * stage.GetTimeCodesPerSecond())), t)
        log["time"].append(t)
        for name, path, _ in cams:
            log[name].append(rc.camera_to_world(stage, path))

    print(f"[rail_demo] live control of {len(cams)} cameras; press Play")
    return omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)


old = getattr(builtins, "_rail_demo_sub", None)
builtins._rail_demo_sub = None  # dropping the subscription unsubscribes it
del old

if MODE != "off":
    # a paused/playing timeline keeps PhysX alive with the rig's old bodies, which write NaN poses back;
    # stop it so the rig is re-parsed without physics
    omni.timeline.get_timeline_interface().stop()
    _stage = omni.usd.get_context().get_stage()
    _cams = make_cameras(_stage)
    if MODE == "live":
        builtins._rail_demo_sub = start_live(_stage, _cams)
    else:
        builtins._rail_demo_task = asyncio.ensure_future(record(_stage, _cams))
