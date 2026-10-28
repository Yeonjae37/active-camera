"""
python scripts/run_pose2sim_demo.py            # fresh copy + run all
python scripts/run_pose2sim_demo.py --keep     # reuse existing copy
python scripts/run_pose2sim_demo.py --skip-pose  # reuse pose/ results
"""

import argparse
import os
import shutil
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import Pose2Sim
from Pose2Sim import Pose2Sim as p2s

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "output" / "pose2sim_demo"
STEPS = ["calibration", "poseEstimation", "synchronization", "personAssociation",
         "triangulation", "filtering", "markerAugmentation", "kinematics"]


def demo_src() -> Path:
    return Path(Pose2Sim.__file__).resolve().parent / "Demo_SinglePerson"


def prepare_project(out_dir: Path, keep: bool) -> Path:
    if out_dir.exists() and not keep:
        shutil.rmtree(out_dir)
    if not out_dir.exists():
        shutil.copytree(demo_src(), out_dir)
    return out_dir


def headless_config(project_dir: Path) -> dict:
    # Overrides merged on top of <project_dir>/Config.toml by Pose2Sim
    return {
        "project": {"project_dir": str(project_dir)},
        "pose": {"display_detection": False},
        "synchronization": {"synchronization_gui": False, "display_sync_plots": False},
        "calibration": {"calculate": {"intrinsics": {"show_detection_intrinsics": False},
                                      "extrinsics": {"show_reprojection_error": False}}},
        "filtering": {"display_figures": False},  # plots still saved (save_filt_plots)
        "kinematics": {"filter_ik": False},  # true would re-enter filtering plots
    }


def list_outputs(project_dir: Path) -> None:
    for pattern in ("calibration/*.toml", "pose-3d/*.trc", "kinematics/*.osim", "kinematics/*.mot"):
        for f in sorted(project_dir.glob(pattern)):
            print(f"  {f.relative_to(project_dir)}  ({f.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep", action="store_true", help="reuse existing copy of the demo")
    ap.add_argument("--skip-pose", action="store_true", help="skip 2D pose estimation (needs pose/ already)")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0],
                    help="resume from this step (use with --keep)")
    args = ap.parse_args()

    project_dir = prepare_project(args.out.resolve(), args.keep or args.skip_pose or args.start != STEPS[0])
    print(f"Pose2Sim {Pose2Sim.__version__ if hasattr(Pose2Sim, '__version__') else ''} demo -> {project_dir}")

    run = {f"do_{s}": i >= STEPS.index(args.start) for i, s in enumerate(STEPS)}
    if args.skip_pose:
        run["do_poseEstimation"] = False
    p2s.runAll(headless_config(project_dir), **run)

    print("\nOutputs:")
    list_outputs(project_dir)


if __name__ == "__main__":
    main()
