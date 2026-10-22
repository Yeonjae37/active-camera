
import omni.ui as ui
import omni.usd
from pxr import Usd, UsdPhysics

PAN_PATH = "/World/pan_tilt/Physics/pan_joint"
TILT_PATH = "/World/pan_tilt/Physics/tilt_joint"

STEP = 5.0

PAN_MIN, PAN_MAX = -180.0, 180.0
TILT_MIN, TILT_MAX = -90.0, 90.0


def set_joint_angle(path, angle):
    stage = omni.usd.get_context().get_stage()
    joint = UsdPhysics.RevoluteJoint.Get(stage, path)

    if not joint:
        raise RuntimeError(f"Joint not found: {path}")

    lower = joint.GetLowerLimitAttr().Get()
    upper = joint.GetUpperLimitAttr().Get()

    if lower is not None and upper is not None and lower <= upper:
        angle = max(float(lower), min(float(upper), angle))

    with Usd.EditContext(stage, stage.GetRootLayer()):
        drive = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular")

        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")

        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(1000.0)
        drive.CreateDampingAttr().Set(100.0)
        drive.CreateMaxForceAttr().Set(1000.0)
        drive.CreateTargetPositionAttr().Set(float(angle))

    print(f"{path.rsplit('/', 1)[-1]} target: {angle:.1f} degrees")


def get_target_angle(path):
    stage = omni.usd.get_context().get_stage()
    joint = UsdPhysics.RevoluteJoint.Get(stage, path)

    if not joint:
        raise RuntimeError(f"Joint not found: {path}")

    drive = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular")
    if not drive:
        return 0.0

    value = drive.GetTargetPositionAttr().Get()
    return float(value) if value is not None else 0.0


class PanTiltController:
    def __init__(self):
        self.window = ui.Window(
            "Pan-Tilt Controller",
            width=380,
            height=280
        )

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Manual Pan-Tilt Control", height=28)

                ui.Label("Pan (degrees)")
                self.pan_slider = ui.FloatSlider(
                    min=PAN_MIN,
                    max=PAN_MAX,
                    step=1.0,
                    height=26
                )

                ui.Label("Tilt (degrees)")
                self.tilt_slider = ui.FloatSlider(
                    min=TILT_MIN,
                    max=TILT_MAX,
                    step=1.0,
                    height=26
                )

                with ui.HStack(height=35, spacing=6):
                    ui.Button(
                        "Pan -5",
                        clicked_fn=lambda: self.change_pan(-STEP)
                    )
                    ui.Button(
                        "Pan +5",
                        clicked_fn=lambda: self.change_pan(STEP)
                    )

                with ui.HStack(height=35, spacing=6):
                    ui.Button(
                        "Tilt -5",
                        clicked_fn=lambda: self.change_tilt(-STEP)
                    )
                    ui.Button(
                        "Tilt +5",
                        clicked_fn=lambda: self.change_tilt(STEP)
                    )

                ui.Button(
                    "Reset to 0",
                    clicked_fn=self.reset,
                    height=35
                )

        # Set initial values before registering callbacks.
        self.pan_slider.model.set_value(get_target_angle(PAN_PATH))
        self.tilt_slider.model.set_value(get_target_angle(TILT_PATH))

        # Keep callback subscriptions alive.
        self.pan_subscription = (
            self.pan_slider.model.subscribe_value_changed_fn(
                self.on_pan_changed
            )
        )

        self.tilt_subscription = (
            self.tilt_slider.model.subscribe_value_changed_fn(
                self.on_tilt_changed
            )
        )

    def on_pan_changed(self, model):
        set_joint_angle(PAN_PATH, model.as_float)

    def on_tilt_changed(self, model):
        set_joint_angle(TILT_PATH, model.as_float)

    def change_pan(self, delta):
        current = self.pan_slider.model.as_float
        self.pan_slider.model.set_value(
            max(PAN_MIN, min(PAN_MAX, current + delta))
        )

    def change_tilt(self, delta):
        current = self.tilt_slider.model.as_float
        self.tilt_slider.model.set_value(
            max(TILT_MIN, min(TILT_MAX, current + delta))
        )

    def reset(self):
        self.pan_slider.model.set_value(0.0)
        self.tilt_slider.model.set_value(0.0)


# Close the previous controller if one exists.
try:
    pan_tilt_controller.window.visible = False
except NameError:
    pass

pan_tilt_controller = PanTiltController()

print("Pan-Tilt Controller ready!")
