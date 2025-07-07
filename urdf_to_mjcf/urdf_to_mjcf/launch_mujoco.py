"""
launch_mujoco.py

Viewer script to load and visualize a MuJoCo MJCF file.
Useful for verifying URDF-to-MJCF conversion results.

Author: Harshavarthan Varatharajan
"""

import mujoco
import mujoco.viewer
import os


def main():
    # Set the path to your MJCF file
    mjcf_path = "/home/hv/robot-model-package-formatter/urdf_to_mjcf/tests/piper_description.xml"

    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"[ERROR] Could not find MJCF file at: {mjcf_path}")

    # Load the MuJoCo model from the XML file
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Launch MuJoCo passive viewer (you manually step the simulation)
    print("[INFO] Launching MuJoCo viewer. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)  # Advance simulation by one step
            viewer.sync()  # Sync state with viewer


if __name__ == "__main__":
    main()
