"""
Interactive preview of the Franka + Panda + tables MuJoCo scene.

Usage::

    python preview.py
"""

import mujoco
import mujoco.viewer
from env.franka_env import FrankaEnv


def main():
    env = FrankaEnv()

    print("Press 'q' or close the window to exit...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat = [0.3, 0, 0.2]

        while viewer.is_running():
            env.step()
            viewer.sync()


if __name__ == "__main__":
    main()
