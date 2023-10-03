import argparse
import os
import sys
import time
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np
import pybullet_data
from movement_primitives.dmp import DMP  # 0.5.0
from pbutils.primitives import PybulletBox, PybulletMesh
from pbutils.robot_interface import PybulletPR2
from pbutils.utils import solve_ik
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_vector
from skrobot.models.pr2 import PR2

import pybullet


# copied from: https://github.com/bulletphysics/bullet3/issues/2170
@contextmanager
def suppress_stdout():

    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class World:
    av_init: np.ndarray
    mesh_pose_init: Coordinates
    ri: PybulletPR2
    pr2: PR2
    cup: PybulletMesh
    box: PybulletBox

    def __init__(self, gui: bool = False):
        if gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setGravity(0.0, 0.0, -9.8)
        pybullet.setTimeStep(0.001)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        with suppress_stdout():
            # loading urdf often prints out annoying warnings
            pybullet.loadURDF("plane.urdf")
            box = PybulletBox(np.array([0.6, 0.8, 0.8]), np.array([0.8, 0.0, 0.4]))
            cup = PybulletMesh(
                Path("./cup_reduce.obj"),
                scale=0.03,
                pos=np.array([0.6, 0.0, 0.8]),
                rot=np.array([np.pi / 2, 0, 0]),
            )

            pr2 = PR2()
            ri = PybulletPR2(pr2)

        pr2.reset_manip_pose()
        pr2.gripper_distance(0.05)
        ri.set_q(pr2.angle_vector(), t_sleep=0.01, simulate=False)

        self.mesh_pose_init = cup.obj.copy_worldcoords()
        self.ri = ri
        self.pr2 = pr2
        self.cup = cup
        self.box = box

        # this come after setting the initial pose of the cup
        solve_ik(pr2, self.co_grasp_pre, sdf=box.sdf)
        ri.set_q(pr2.angle_vector(), t_sleep=0.0, simulate=False)
        self.av_init = pr2.angle_vector()
        self.relative_grasping_trajectory  # cached property compute cache now

    @property
    def co_handle(self) -> Coordinates:
        co_handle = self.cup.obj.copy_worldcoords()
        co_handle.translate([0, 0, 0.07])
        co_handle.rotate(np.pi * 0.2, "z")
        co_handle.translate([-0.12, -0.0, 0.0])
        co_handle.rotate(-np.pi * 0.5, "z")
        return co_handle

    @property
    def co_grasp(self) -> Coordinates:
        co_grasp = self.co_handle.copy_worldcoords()
        co_grasp.rotate(0.3, "z")
        co_grasp.translate([0.02, -0.015, 0.0])
        return co_grasp

    @property
    def co_grasp_pre(self) -> Coordinates:
        co_grasp_pre = self.co_grasp.copy_worldcoords()
        co_grasp_pre.translate([-0.1, -0.0, 0.0])
        return co_grasp_pre

    def set_cup_position_offset(self, offset: np.ndarray, angle: float = 0.0) -> None:
        co = self.mesh_pose_init.copy_worldcoords()
        co.translate(offset, wrt="world")
        co.rotate(angle, "x")
        self.cup.set_coords(co)

    def reset(self) -> None:
        self.pr2.angle_vector(self.av_init)
        self.ri.set_q(self.av_init, t_sleep=0.0, simulate=False)
        self.cup.set_coords(self.mesh_pose_init)

    @cached_property
    def relative_grasping_trajectory(self) -> DMP:
        co_grasp = self.co_grasp.copy_worldcoords()
        n_split = 100

        traj_co = [co_grasp.copy_worldcoords().transform(self.co_handle.inverse_transformation())]
        slide_per_dt = 0.1 / n_split
        for _ in range(n_split - 1):
            co_grasp.translate([-slide_per_dt, 0, 0])
            co_grasp_wrt_handle = co_grasp.copy_worldcoords().transform(
                self.co_handle.inverse_transformation()
            )
            traj_co.append(co_grasp_wrt_handle)

        traj_co = traj_co[::-1]
        traj_xyzquat = np.array([np.hstack([co.worldpos(), co.quaternion]) for co in traj_co])

        times = np.linspace(0, 1, n_split)
        dmp = DMP(7, execution_time=1.0, n_weights_per_dim=10, dt=0.1)
        dmp.imitate(times, np.array(traj_xyzquat))
        return dmp

    def reproduce_grasping_trajectory(self, dmp: DMP) -> None:
        co_rarm_wrt_world = self.pr2.rarm_end_coords.copy_worldcoords()
        co_rarm_wrt_handle = co_rarm_wrt_world.copy_worldcoords().transform(
            self.co_handle.inverse_transformation()
        )
        xyzquat_start = np.hstack([co_rarm_wrt_handle.worldpos(), co_rarm_wrt_handle.quaternion])
        dmp.reset()
        dmp.configure(start_y=xyzquat_start)
        T, xyzquat_list = dmp.open_loop()

        for xyzquat in xyzquat_list:
            q = normalize_vector(xyzquat[3:])
            co_rarm = Coordinates(xyzquat[:3], q)
            co_rarm = co_rarm.transform(self.co_handle)
            solve_ik(self.pr2, co_rarm)
            self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    world = World(gui=args.gui)
    world.reset()
    print(world.co_handle)
    world.set_cup_position_offset(np.array([+0.12, -0.0, 0.0]), -0.0)
    print(world.co_handle)
    world.reproduce_grasping_trajectory(world.relative_grasping_trajectory)
    time.sleep(1.0)
    # world.reset()
    import time

    time.sleep(1000)
