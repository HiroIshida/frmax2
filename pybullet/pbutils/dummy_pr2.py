import os

import numpy as np
from skrobot.coordinates import CascadedCoords
from skrobot.models.urdf import RobotModelFromURDF


class DummyPR2(RobotModelFromURDF):
    def __init__(self):
        super().__init__(urdf_file=self.default_urdf_path)
        self.rarm_end_coords = CascadedCoords(
            parent=self.r_gripper_tool_frame, name="rarm_end_coords"
        )

    @property
    def default_urdf_path(self) -> str:
        # gripper.urdf in this directory
        return os.path.join(os.path.dirname(__file__), "gripper.urdf")

    def reset_manip_pose(self) -> None:
        ...

    def gripper_distance(self, dist=None, arm="arms"):
        # dont care arm
        joints = [self.r_gripper_l_finger_joint]

        def _dist(angle):
            return 0.0099 * (18.4586 * np.sin(angle) + np.cos(angle) - 1.0101)

        if dist is not None:
            # calculate joint_angle from approximated equation
            max_dist = _dist(joints[0].max_angle)
            dist = max(min(dist, max_dist), 0)
            d = dist / 2.0
            angle = 2 * np.arctan(
                (9137 - np.sqrt(2) * np.sqrt(-5e9 * (d**2) - 5e7 * d + 41739897))
                / (5 * (20000 * d + 199))
            )
            for joint in joints:
                joint.joint_angle(angle)
        angle = joints[0].joint_angle()
        return _dist(angle)
