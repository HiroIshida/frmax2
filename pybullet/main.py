import argparse
import os
import sys
import time
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet_data
from movement_primitives.dmp import CartesianDMP
from pbutils.primitives import PybulletBox, PybulletMesh
from pbutils.robot_interface import PybulletPR2
from pbutils.utils import solve_ik
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_vector
from skrobot.models.pr2 import PR2
from utils import CoordinateTransform, chain_transform

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


def create_debug_axis(coords: Coordinates, length: float = 0.1):
    start = coords.worldpos()
    end_x = start + coords.rotate_vector([length, 0, 0])
    pybullet.addUserDebugLine(start, end_x, [1, 0, 0], 3)
    end_y = start + coords.rotate_vector([0, length, 0])
    pybullet.addUserDebugLine(start, end_y, [0, 1, 0], 3)
    end_z = start + coords.rotate_vector([0, 0, length])
    pybullet.addUserDebugLine(start, end_z, [0, 0, 1], 3)


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
            # if True:
            # loading urdf often prints out annoying warnings
            pybullet.loadURDF("plane.urdf")
            box = PybulletBox(np.array([0.6, 0.8, 0.8]), np.array([0.8, 0.0, 0.4]))
            cup = PybulletMesh(
                Path("./cup_reduce.obj"),
                scale=0.03,
                pos=np.array([0.6, 0.0, 0.8]),
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
        self.relative_grasping_dmp  # cached property compute cache now

    @property
    def co_handle(self) -> Coordinates:
        co_handle = self.cup.obj.copy_worldcoords()
        co_handle.translate([0, 0, 0.07])
        co_handle.rotate(np.pi * 0.2, "z")
        co_handle.translate([0.0, -0.12, 0.0])
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
        co.rotate(angle, "z", wrt="world")
        self.cup.set_coords(co)

    def reset(self) -> None:
        self.pr2.angle_vector(self.av_init)
        self.ri.set_q(self.av_init, t_sleep=0.0, simulate=False)
        self.cup.set_coords(self.mesh_pose_init)

    def get_wrt_handle(
        self, co_grasp2world: Coordinates, recog_error: Optional[np.ndarray] = None
    ) -> Coordinates:
        if recog_error is None:
            recog_error = np.zeros(3)

        co_handle_error = self.co_handle.copy_worldcoords()
        trans = np.zeros(3)
        trans[:2] = recog_error[:2]
        co_handle_error.translate(trans, wrt="world")
        co_handle_error.rotate(recog_error[2], "z", wrt="local")

        tf_g2w = CoordinateTransform.from_skrobot_coords(co_grasp2world, "g", "w")
        tf_h2w = CoordinateTransform.from_skrobot_coords(co_handle_error, "h", "w")
        tf_g2h = chain_transform(tf_g2w, tf_h2w.inverse())
        return tf_g2h.to_skrobot_coords()

    def get_wrt_world(
        self, co_gripper2handle: Coordinates, recog_error: Optional[np.ndarray] = None
    ) -> Coordinates:
        if recog_error is None:
            recog_error = np.zeros(3)
        co_handle_error = self.co_handle.copy_worldcoords()
        trans = np.zeros(3)
        trans[:2] = recog_error[:2]
        co_handle_error.translate(trans, wrt="world")
        co_handle_error.rotate(recog_error[2], "z", wrt="local")

        tf_g2h = CoordinateTransform.from_skrobot_coords(co_gripper2handle, "g", "h")
        tf_h2w = CoordinateTransform.from_skrobot_coords(co_handle_error, "h", "w")
        tf_g2w = chain_transform(tf_g2h, tf_h2w)
        return tf_g2w.to_skrobot_coords()

    @cached_property
    def relative_grasping_dmp(self) -> CartesianDMP:
        co_grasp = self.co_grasp.copy_worldcoords()
        n_split = 100
        traj_co = [self.get_wrt_handle(co_grasp)]
        slide_per_dt = 0.1 / n_split
        for _ in range(n_split - 1):
            co_grasp.translate([-slide_per_dt, 0, 0])
            traj_co.append(self.get_wrt_handle(co_grasp))

        traj_co = traj_co[::-1]
        traj_xyzquat = np.array([np.hstack([co.worldpos(), co.quaternion]) for co in traj_co])

        times = np.linspace(0, 1, n_split)
        dmp = CartesianDMP(execution_time=1.0, n_weights_per_dim=10, dt=0.1)
        dmp.imitate(times, np.array(traj_xyzquat))
        return dmp

    def reproduce_grasping_dmp(
        self, dmp: CartesianDMP, recog_error: Optional[np.ndarray] = None
    ) -> None:
        if recog_error is None:
            recog_error = np.zeros(3)

        co_rarm_wrt_world = self.pr2.rarm_end_coords.copy_worldcoords()
        co_rarm_wrt_handle = self.get_wrt_handle(co_rarm_wrt_world, recog_error)

        xyzquat_start = np.hstack([co_rarm_wrt_handle.worldpos(), co_rarm_wrt_handle.quaternion])
        dmp.reset()
        dmp.configure(start_y=xyzquat_start)
        T, xyzquat_list = dmp.open_loop()

        traj = []
        for xyzquat in xyzquat_list:
            q = normalize_vector(xyzquat[3:])
            co_rarm2handle = Coordinates(xyzquat[:3], q)
            co_rarm2world = self.get_wrt_world(co_rarm2handle, recog_error)
            traj.append(co_rarm2world)

        for co_rarm2world in traj:
            create_debug_axis(co_rarm2world)

        for co_rarm2world in traj:
            solve_ik(self.pr2, co_rarm2world)
            self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)

        # finally grasp
        self.pr2.gripper_distance(0.01)
        self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)

    def check_grasp_success(self) -> bool:
        co_rarm_wrt_world = self.pr2.rarm_end_coords.copy_worldcoords()
        co_rarm_wrt_world.translate([0, 0, 0.05])
        solve_ik(self.pr2, co_rarm_wrt_world)

        self.cup.sync()
        pos_pre_lift = self.cup.obj.worldpos()

        self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)
        self.cup.sync()
        pos_post_lift = self.cup.obj.worldpos()
        return np.linalg.norm(pos_pre_lift - pos_post_lift) > 0.03


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    world = World(gui=not args.nogui)
    world.reset()
    create_debug_axis(world.co_handle)
    world.set_cup_position_offset(np.zeros(3), +0.3)
    dmp = world.relative_grasping_dmp
    W = dmp.forcing_term_pos.weights
    dmp.forcing_term_pos.weights[:2, :] += np.random.randn(*W.shape)[:2, :] * 50
    world.reproduce_grasping_dmp(dmp, np.array([0.0, 0.0, -0.0]))
    assert world.check_grasp_success()
    world.reset()
    time.sleep(1000)
