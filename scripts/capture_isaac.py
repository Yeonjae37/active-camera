
import asyncio
import csv
from datetime import datetime
from pathlib import Path

import omni.timeline
from omni.kit.viewport.utility import (
    get_active_viewport,
    capture_viewport_to_file,
    next_viewport_frame_async,
)

ROOT = Path.home() / "Study/active-camera"

DURATION = 10.0
NUM_FRAMES = 30

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = ROOT / "output" / "walking_capture" / RUN_ID


async def capture_walking():
    viewport = get_active_viewport()
    timeline = omni.timeline.get_timeline_interface()

    if viewport is None:
        raise RuntimeError("Active viewport not found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timeline.stop()
    timeline.set_looping(False)
    timeline.set_current_time(0.0)

    # Do not capture beyond the animation's end time.
    end_time = min(DURATION, timeline.get_end_time())

    if end_time <= 0:
        raise RuntimeError("Invalid timeline end time")

    print("Camera:", viewport.camera_path)
    print("Capture duration:", end_time)
    print("Target frames:", NUM_FRAMES)

    # Capture at evenly spaced animation times.
    target_times = [
        (i + 0.5) * end_time / NUM_FRAMES
        for i in range(NUM_FRAMES)
    ]

    records = []

    try:
        timeline.play()

        for i, target_time in enumerate(target_times):

            # Advance the actual walking animation.
            while timeline.get_current_time() < target_time:
                if not timeline.is_playing():
                    raise RuntimeError(
                        "Timeline stopped before target time"
                    )

                await next_viewport_frame_async(
                    viewport, n_frames=1
                )

            # Freeze the walking animation during PNG capture.
            timeline.pause()

            # Allow the viewport to render the paused pose.
            await next_viewport_frame_async(
                viewport, n_frames=2
            )

            actual_time = timeline.get_current_time()
            image_path = OUTPUT_DIR / f"{i:06d}.png"

            capture = capture_viewport_to_file(
                viewport,
                file_path=str(image_path),
            )

            await capture.wait_for_result(
                completion_frames=60
            )

            records.append((
                i,
                target_time,
                actual_time,
                str(image_path),
            ))

            print(
                f"Captured {i + 1:02d}/{NUM_FRAMES} "
                f"| target={target_time:.3f}s "
                f"| actual={actual_time:.3f}s",
                flush=True,
            )

            # Resume only after the image has been saved.
            if i < NUM_FRAMES - 1:
                timeline.play()
                await next_viewport_frame_async(
                    viewport, n_frames=1
                )

        csv_path = OUTPUT_DIR / "timestamps.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "target_time_sec",
                "actual_time_sec",
                "image_path",
            ])
            writer.writerows(records)

        print("\nCapture completed!")
        print("Saved frames:", len(records))
        print("Output:", OUTPUT_DIR)

    finally:
        timeline.pause()


asyncio.ensure_future(capture_walking())
