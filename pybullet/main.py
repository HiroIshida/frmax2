import argparse
import copy
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional, Union, overload

import dill
import matplotlib.pyplot as plt
import numpy as np
import pybullet_data
import tqdm
from frmax.initialize import initialize
from movement_primitives.dmp import DMP as _DMP
from movement_primitives.dmp import CartesianDMP as _CartesianDMP
from pbutils.primitives import PybulletBox, PybulletMesh
from pbutils.robot_interface import PybulletPR2
from pbutils.utils import solve_ik, solve_ik_optimization
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_vector
from skrobot.models.pr2 import PR2
from utils import CoordinateTransform, chain_transform

import pybullet
from frmax2.core import ActiveSamplerConfig, HolllessActiveSampler
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger

logger = logging.getLogger(__name__)


class CartesianDMP(_CartesianDMP):
    def set_param(self, param: np.ndarray) -> None:
        dof = self.n_weights_per_dim * 5 + 2  # +2 for goal parameter currently only for 2d pos
        assert len(param) == dof
        W = param[:-2].reshape(5, self.n_weights_per_dim)
        self.forcing_term_pos.weights[:2, :] = W[:2, :]
        self.forcing_term_rot.weights = W[2:, :]
        goal_param = param[-2:]
        self.goal_y[:2] = goal_param


class DMP(_DMP):
    def set_param(self, param: np.ndarray) -> None:
        n_dim = 3
        n_goal_dim = 3
        dof = self.n_weights_per_dim * n_dim + n_goal_dim
        assert len(param) == dof
        W = param[:-n_goal_dim].reshape(n_dim, self.n_weights_per_dim)
        self.forcing_term.weights[:2, :] += W[:2, :]
        self.forcing_term.weights[3, :] += W[2, :]

        # set goal param
        goal_param = param[-n_goal_dim:]
        self.goal_y[:2] += goal_param[:2]
        self.goal_y[3] += goal_param[2]


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
        solve_ik_optimization(pr2, self.co_grasp_pre(), sdf=box.sdf)
        ri.set_q(pr2.angle_vector(), t_sleep=0.0, simulate=False)
        self.av_init = pr2.angle_vector()

    def rollout(self, param: np.ndarray, recog_error: np.ndarray) -> bool:
        is_pos_only = len(recog_error) == 2
        assert is_pos_only
        if is_pos_only:
            recog_error = np.hstack([recog_error, 0.0])

        dmp = copy.deepcopy(self.get_relative_grasping_dmp())
        dmp.set_param(param)
        self.reproduce_grasping_dmp(dmp, recog_error, True)
        ret = self.check_grasp_success()
        self.reset()
        return ret

    @property
    def co_handle(self) -> Coordinates:
        co_handle = self.cup.obj.copy_worldcoords()
        co_handle.translate([0, 0, 0.07])
        co_handle.rotate(np.pi * 0.2, "z")
        co_handle.translate([0.0, -0.12, 0.0])
        return co_handle

    def co_grasp(self, co_handle_recog: Optional[Coordinates] = None) -> Coordinates:
        if co_handle_recog is None:
            co_handle_recog = self.co_handle
        co_grasp = co_handle_recog.copy_worldcoords()
        co_grasp.rotate(0.3, "z")
        co_grasp.translate([0.02, -0.015, 0.0])
        return co_grasp

    def co_grasp_pre(self, co_handle_recog: Optional[Coordinates] = None) -> Coordinates:
        co_grasp_pre = self.co_grasp(co_handle_recog).copy_worldcoords()
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

    @overload
    def get_relative_grasping_dmp(self, mode: Literal["quat"]) -> CartesianDMP:
        ...

    @overload
    def get_relative_grasping_dmp(self, mode: Literal["yaw"]) -> DMP:
        ...

    def get_relative_grasping_dmp(self, mode: Literal["quat", "yaw"] = "yaw"):
        co_grasp = self.co_grasp().copy_worldcoords()
        n_split = 100
        traj_co = [self.get_wrt_handle(co_grasp)]
        slide_per_dt = 0.1 / n_split
        for _ in range(n_split - 1):
            co_grasp.translate([-slide_per_dt, 0, 0])
            traj_co.append(self.get_wrt_handle(co_grasp))

        traj_co = traj_co[::-1]
        times = np.linspace(0, 1, n_split)

        if mode == "quat":
            traj_xyzquat = np.array([np.hstack([co.worldpos(), co.quaternion]) for co in traj_co])
            dmp = CartesianDMP(execution_time=1.0, n_weights_per_dim=10, dt=0.1)
            dmp.imitate(times, np.array(traj_xyzquat))
        else:
            traj_xyzyaw = []
            for co in traj_co:
                yaw = co.rpy_angle()[0][0]
                vec = np.hstack([co.worldpos(), yaw])
                traj_xyzyaw.append(vec)
            traj_xyzyaw = np.array(traj_xyzyaw)
            dmp = DMP(4, execution_time=1.0, n_weights_per_dim=10, dt=0.1)
            dmp.imitate(times, np.array(traj_xyzyaw))
        return dmp

    def reproduce_grasping_dmp(
        self,
        dmp: Union[CartesianDMP, DMP],
        recog_error: Optional[np.ndarray] = None,
        show_debug_axis: bool = False,
    ) -> None:
        if recog_error is None:
            recog_error = np.zeros(3)

        co_rarm_wrt_world = self.pr2.rarm_end_coords.copy_worldcoords()
        co_rarm_wrt_handle = self.get_wrt_handle(co_rarm_wrt_world, recog_error)

        if isinstance(dmp, CartesianDMP):
            xyzquat_start = np.hstack(
                [co_rarm_wrt_handle.worldpos(), co_rarm_wrt_handle.quaternion]
            )
            dmp.reset()
            dmp.configure(start_y=xyzquat_start)
            T, xyzquat_list = dmp.open_loop()

            traj = []
            for xyzquat in xyzquat_list:
                q = normalize_vector(xyzquat[3:])
                co_rarm2handle = Coordinates(xyzquat[:3], q)
                co_rarm2world = self.get_wrt_world(co_rarm2handle, recog_error)
                traj.append(co_rarm2world)
        else:
            xyzyaw = np.hstack(
                [co_rarm_wrt_handle.worldpos(), co_rarm_wrt_handle.rpy_angle()[0][0]]
            )
            dmp.reset()
            dmp.configure(start_y=xyzyaw)
            T, xyzyaw_list = dmp.open_loop()

            traj = []
            for xyzyaw in xyzyaw_list:
                co_rarm2handle = Coordinates(xyzyaw[:3])
                co_rarm2handle.rotate(xyzyaw[3], "z", wrt="local")
                co_rarm2world = self.get_wrt_world(co_rarm2handle, recog_error)
                traj.append(co_rarm2world)

        pybullet.removeAllUserDebugItems()
        if show_debug_axis:
            for co_rarm2world in traj:
                create_debug_axis(co_rarm2world)

        for co_rarm2world in traj:
            if not solve_ik(self.pr2, co_rarm2world):
                logger.error("ik failed in reproduce_grasping_dmp")
            self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)

        # finally grasp
        self.pr2.gripper_distance(0.01)
        self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)

    def check_grasp_success(self) -> bool:
        co_rarm_wrt_world = self.pr2.rarm_end_coords.copy_worldcoords()
        co_rarm_wrt_world.translate([0, 0, 0.05])
        if not solve_ik(self.pr2, co_rarm_wrt_world):
            logger.error("ik failed in check_grasp_success")

        self.cup.sync()
        pos_pre_lift = self.cup.obj.worldpos()

        self.ri.set_q(self.pr2.angle_vector(), t_sleep=0.0, simulate=True)
        self.cup.sync()
        pos_post_lift = self.cup.obj.worldpos()
        return np.linalg.norm(pos_pre_lift - pos_post_lift) > 0.03


def _process_pool_setup():
    global _pp_world
    _pp_world = World(gui=False)
    _pp_world.reset()


def _process_pool_rollout(param: np.ndarray, error: np.ndarray) -> bool:
    global _pp_world
    return _pp_world.rollout(param, error)


class RobustGraspTrainer:
    param_dim: int
    config: ActiveSamplerConfig
    sampler: HolllessActiveSampler
    is_valid_error: Callable[[np.ndarray], bool]

    def __init__(self, world: World):
        self.world = world

        def is_valid_error(err: np.ndarray) -> bool:
            return np.linalg.norm(err) < 0.1

        self.is_valid_error = is_valid_error

        error_dim = 2
        dmp_param_dim = 30
        goal_param_dim = 3
        self.param_dim = dmp_param_dim + goal_param_dim

        # ls_param = np.hstack([100 * np.ones(dmp_param_dim), [0.01, 0.01, 0.1]])
        ls_param = np.hstack([100 * np.ones(dmp_param_dim), [0.06, 0.06, 0.6]])
        ls_error = 0.3 * np.ones(2)
        param_init = np.zeros(self.param_dim)

        # check if at least grasp succssfull with initial param and error 0
        x = np.hstack([param_init, np.zeros(error_dim)])
        assert self.rollout(x)

        X, Y, ls_error = initialize(
            lambda x: +1 if self.rollout(x) else -1, param_init, ls_error, eps=0.05
        )

        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        config = ActiveSamplerConfig()
        sampler = HolllessActiveSampler(X, Y, metric, param_init, config)
        self.config = config
        self.sampler = sampler

    def rollout(self, x: np.ndarray) -> bool:
        param = x[: self.param_dim]
        error = x[self.param_dim :]

        goal_param = param[-3:]
        logger.info(f"rollout with goal param: {goal_param} and error: {error}")
        logger.debug(f"rollout param: {param}")
        ret = False
        if self.is_valid_error(error):
            ret = self.world.rollout(param, error)
        logger.info(f"rollout result: {ret}")
        return ret

    def train(self, n_iter) -> None:
        for i in range(n_iter):
            print(f"iter: {i}")
            x = self.sampler.ask()
            self.sampler.tell(x, self.rollout(x))
            with open("cache.pkl", "wb") as f:
                dill.dump(self.sampler, f)


def load_sampler() -> HolllessActiveSampler:
    with open("cache.pkl", "rb") as f:
        sampler = dill.load(f)
    return sampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode not in ("plot", "eval"):
        np.random.seed(args.seed)
        world = World(gui=not args.nogui)
        world.reset()

        if args.mode == "debug":
            dmp = world.get_relative_grasping_dmp()
            dmp.set_param(np.zeros(33))
            world.reproduce_grasping_dmp(dmp, np.array([0.0, 0.0, -0.0]))
            assert world.check_grasp_success()
            world.reset()
            time.sleep(1000)
        elif args.mode == "train":
            create_default_logger(Path("./"), "train", logging.INFO)
            trainer = RobustGraspTrainer(world)
            trainer.train(args.n)
        elif args.mode == "test":
            sampler = load_sampler()
            param = sampler.best_param_so_far
            dmp = copy.deepcopy(world.get_relative_grasping_dmp())
            dmp.set_param(param)
            world.reproduce_grasping_dmp(dmp, np.array([0.0, -0.0, 0.0]))
            world.check_grasp_success()
            time.sleep(1000)
        else:
            assert False
    else:
        if args.mode == "plot":
            sampler = load_sampler()
            fig, ax = plt.subplots()
            ax.plot(sampler.sampler_cache.best_volume_history)
            plt.show()
        elif args.mode == "eval":

            sampler = load_sampler()
            param = sampler.best_param_so_far

            fig, ax = plt.subplots()
            sampler.fslset.show_sliced(param, list(range(len(param))), 30, (fig, ax))
            ax.set_xlim([-0.15, 0.15])
            ax.set_ylim([-0.15, 0.15])

            error_x_lin = np.linspace(-0.1, 0.1, 10)
            error_y_lin = np.linspace(-0.1, 0.1, 10)
            error_x, error_y = np.meshgrid(error_x_lin, error_y_lin)
            pts = np.vstack([error_x.flatten(), error_y.flatten()]).T
            # bools = []
            # world = World(gui=not args.nogui)
            # world.reset()
            # for error in tqdm.tqdm(pts):
            #     ret = world.rollout(param, error)
            #     bools.append(ret)
            with ProcessPoolExecutor(max_workers=8, initializer=_process_pool_setup) as executor:
                bools = list(
                    tqdm.tqdm(
                        executor.map(_process_pool_rollout, [param] * len(pts), pts),
                        total=len(pts),
                    )
                )
            ax.scatter(pts[:, 0], pts[:, 1], c=bools)
            plt.show()
        else:
            assert False
