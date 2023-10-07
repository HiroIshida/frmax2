import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List

import numpy as np
from skmp.robot.utils import get_robot_state
from skrobot.model.link import Link
from skrobot.models.pr2 import PR2
from tinyfk import BaseType

import pybullet


class PybulletPR2Base(ABC):
    pr2: PR2
    robot_id: int
    joint_name_to_id_table: Dict[str, int]
    joint_id_to_name_table: Dict[int, str]
    link_name_to_id_table: Dict[str, int]
    link_id_to_name_table: Dict[int, str]
    max_angle_diff_list: List[float]  # same order as joint_lst
    with_base: bool

    def __init__(self, pr2: PR2, with_base: bool = False):
        self.pr2 = pr2
        robot_id = pybullet.loadURDF(pr2.default_urdf_path, useFixedBase=not with_base)
        self.robot_id = robot_id

        link_name_to_id_table = {pybullet.getBodyInfo(robot_id)[0].decode("UTF-8"): -1}
        joint_name_to_id_table = {}

        def heck(path: str) -> str:
            return "_".join(path.split("/"))

        for _id in range(pybullet.getNumJoints(robot_id)):
            joint_info = pybullet.getJointInfo(robot_id, _id)
            joint_id = joint_info[0]

            joint_name = joint_info[1].decode("UTF-8")
            joint_name_to_id_table[joint_name] = joint_id
            name_ = joint_info[12].decode("UTF-8")
            name = heck(name_)
            link_name_to_id_table[name] = _id
        joint_id_to_name_table = {v: k for k, v in joint_name_to_id_table.items()}
        link_id_to_name_table = {v: k for k, v in link_name_to_id_table.items()}

        self.joint_name_to_id_table = joint_name_to_id_table
        self.joint_id_to_name_table = joint_id_to_name_table
        self.link_name_to_id_table = link_name_to_id_table
        self.link_id_to_name_table = link_id_to_name_table
        self.with_base = with_base
        self.max_angle_diff_list = self.compute_max_angle_diff_list()  # must call at last

    @abstractmethod
    def compute_max_angle_diff_list(self) -> List[float]:
        ...

    def is_in_collision(self, object_id: int) -> bool:
        ret = pybullet.getContactPoints(bodyA=self.robot_id, bodyB=object_id)
        return len(ret) > 0

    @cached_property
    def pb_joint_ids(self) -> List[int]:
        pb_joint_ids = []
        for joint in self.pr2.joint_list:
            pb_joint_id = self.joint_name_to_id_table[joint.name]
            pb_joint_ids.append(pb_joint_id)
        return pb_joint_ids

    def get_q(self) -> np.ndarray:
        pb_current_angle = []
        for pb_joint_id in self.pb_joint_ids:
            pb_angle = pybullet.getJointState(self.robot_id, pb_joint_id)[0]
            pb_current_angle.append(pb_angle)
        pb_current_angle = np.array(pb_current_angle)

        if self.with_base:
            base_pos, base_quat = pybullet.getBasePositionAndOrientation(self.robot_id)
            base_rpy = pybullet.getEulerFromQuaternion(base_quat)
            pb_current_angle = np.hstack([pb_current_angle, base_pos, base_rpy])

        return pb_current_angle

    def _set_q(self, q: np.ndarray, simulate_lower: bool = True) -> None:
        if self.with_base:
            q, base_pos, base_rpy = q[:-6], q[-6:-3], q[-3:]
            base_quat = pybullet.getQuaternionFromEuler(base_rpy)
            pybullet.resetBasePositionAndOrientation(self.robot_id, base_pos, base_quat)

        for pb_joint_id, angle in zip(self.pb_joint_ids, q):
            pybullet.resetJointState(self.robot_id, pb_joint_id, angle)

        if simulate_lower:
            pybullet.stepSimulation()
            while not self.is_environment_static():
                pybullet.stepSimulation()

    def set_skrobot_state(
        self,
        robot_model: PR2,
        simulate: bool = True,
        simulate_lower: bool = True,
        t_sleep: float = 0.0,
    ) -> None:
        assert robot_model is self.pr2
        base_type = BaseType.FLOATING if self.with_base else BaseType.FIXED
        q = get_robot_state(robot_model, robot_model.joint_names, base_type=base_type)
        self.set_q(q, simulate=simulate, simulate_lower=simulate_lower, t_sleep=t_sleep)

    def set_q(
        self,
        q_target: np.ndarray,
        simulate: bool = True,
        simulate_lower: bool = True,
        t_sleep: float = 0.0,
    ) -> None:
        if simulate:
            q_current = self.get_q()
            diff = q_target - q_current

            largest_index = np.argmax(np.abs(diff) / np.array(self.max_angle_diff_list))

            max_diff = np.abs(diff)[largest_index]
            max_angle_diff = self.max_angle_diff_list[largest_index]
            phase_per_step = max_angle_diff / max_diff
            if phase_per_step > 1.0:
                phase_list = [0.0]
            else:
                phase_list = [phase_per_step * i for i in range(int(max_diff // max_angle_diff))]
            phase_list.append(1.0)
            assert phase_list == sorted(phase_list)

            waypoints = [q_current + diff * phase for phase in phase_list]
            np.testing.assert_almost_equal(waypoints[-1], q_target)
            for av in waypoints[1:]:
                self._set_q(av, simulate_lower=simulate_lower)
                time.sleep(t_sleep)
        else:
            self._set_q(q_target, simulate_lower=simulate_lower)

    def is_environment_static(self):
        bodies = pybullet.getNumBodies()
        for body_id in range(bodies):
            if body_id == self.robot_id:
                continue
            lin_vel, ang_vel = pybullet.getBaseVelocity(body_id)
            if np.linalg.norm(lin_vel) > 0.05 or np.linalg.norm(ang_vel) > 0.05:
                return False
        return True


class PybulletPR2(PybulletPR2Base):
    def compute_max_angle_diff_list(self) -> List[float]:
        max_angle_diff_table = {name: 0.005 for name in self.joint_name_to_id_table.keys()}

        # joint close to end effector should have larger max_angle_diff
        # first get child joints from r_forarm_link
        close_joint_names = []

        def recursion(link: Link):
            clinks = link.child_links
            if len(clinks) == 0:
                return
            for clink in clinks:
                close_joint_names.append(clink.joint.name)
                recursion(clink)

        recursion(self.pr2.r_forearm_link)
        recursion(self.pr2.l_forearm_link)
        for joint_name in close_joint_names:
            max_angle_diff_table[joint_name] = 0.02
        max_angle_diff_list = [max_angle_diff_table[joint.name] for joint in self.pr2.joint_list]

        if self.with_base:
            max_angle_diff_list += [np.min(max_angle_diff_list)] * 6
        return max_angle_diff_list


class PybulletDummyPR2(PybulletPR2Base):
    def compute_max_angle_diff_list(self) -> List[float]:
        max_angle_diff_table = {name: 0.001 for name in self.joint_name_to_id_table.keys()}
        for joint_name, value in max_angle_diff_table.items():
            if "gripper" in joint_name:
                max_angle_diff_table[joint_name] = 0.02
        max_angle_diff_list = [max_angle_diff_table[joint.name] for joint in self.pr2.joint_list]
        if self.with_base:
            max_angle_diff_list += [np.min(max_angle_diff_list)] * 6
        return max_angle_diff_list


class PybulletGripperOnly(PybulletPR2Base):
    def compute_max_angle_diff_list(self) -> List[float]:
        max_angle_diff_table = {name: 0.001 for name in self.joint_name_to_id_table.keys()}
        for joint_name, value in max_angle_diff_table.items():
            if "gripper" in joint_name:
                max_angle_diff_table[joint_name] = 0.02
        max_angle_diff_list = [max_angle_diff_table[joint.name] for joint in self.pr2.joint_list]
        if self.with_base:
            max_angle_diff_list += [0.001] * 6
        return max_angle_diff_list
