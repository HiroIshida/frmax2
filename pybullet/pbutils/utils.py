import logging
from typing import Callable, Optional

from skmp.constraint import CollFreeConst, PoseConstraint
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import satisfy_by_optimization, satisfy_by_optimization_with_budget
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2

logger = logging.getLogger(__name__)


def solve_ik(
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
        logger.error(f"ik failed. attempted coordinate is {co}")
        return False
    set_robot_state(pr2, joint_list, res.q)
    return True
