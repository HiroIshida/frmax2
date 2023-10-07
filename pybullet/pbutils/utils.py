import logging
from typing import Callable, Optional

from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import satisfy_by_optimization, satisfy_by_optimization_with_budget
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2

from .dummy_pr2 import DummyPR2, GripperOnly

logger = logging.getLogger(__name__)


def solve_ik(pr2: PR2, co: Coordinates) -> bool:
    if isinstance(pr2, GripperOnly):
        co = co.copy_worldcoords()
        co.translate([-0.08, 0.0, 0.0])
        pr2.newcoords(co)
        return True  # always solvable

    if isinstance(pr2, DummyPR2):
        link_list = None
    elif isinstance(pr2, DummyPR2):
        link_list = pr2.rarm.link_list
    ret = pr2.inverse_kinematics(
        co,
        link_list=link_list,
        move_target=pr2.rarm_end_coords,
        rotation_axis=True,
        stop=100,
    )
    return ret is not False  # ret may be angle


def solve_ik_optimization(
    pr2: PR2, co: Coordinates, random_sampling: bool = False, sdf: Optional[Callable] = None
) -> bool:
    pr2_conf = PR2Config()
    colkin = pr2_conf.get_collision_kin()
    efkin = pr2_conf.get_endeffector_kin()
    efkin.reflect_skrobot_model(pr2)
    colkin.reflect_skrobot_model(pr2)

    box_const = pr2_conf.get_box_const()
    if sdf is not None:
        collfree_const = CollFreeConst(colkin, sdf, pr2)
    else:
        collfree_const = None
    goal_eq_const = PoseConstraint.from_skrobot_coords([co], efkin, pr2)
    joint_list = pr2_conf._get_control_joint_names()
    q_start = get_robot_state(pr2, joint_list)

    if random_sampling:
        res = satisfy_by_optimization_with_budget(goal_eq_const, box_const, collfree_const, q_start)
    else:
        res = satisfy_by_optimization(goal_eq_const, box_const, collfree_const, q_start)
    if not res.success:
        return False
    set_robot_state(pr2, joint_list, res.q)
    return True
