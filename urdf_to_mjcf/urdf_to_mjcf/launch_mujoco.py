import mujoco
import mujoco.viewer
import os

# Path to your MJCF XML
mjcf_path = "/home/hv/robot-model-package-formatter/urdf_to_mjcf/tests/piper_description.xml"  # Or the full path if not in same directory


if not os.path.isfile(mjcf_path):
    raise FileNotFoundError(f"Could not find MJCF file at {mjcf_path}")

# Load the model
model = mujoco.MjModel.from_xml_path(mjcf_path)

# Create data
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo viewer launched. Close the window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
