from dataclasses import dataclass
from typing import Optional

import numpy as np
from skrobot.coordinates import Coordinates


@dataclass
class CoordinateTransform:
    trans: np.ndarray
    rot: np.ndarray
    src: Optional[str] = None
    dest: Optional[str] = None

    def __call__(self, vec_src: np.ndarray) -> np.ndarray:
        if vec_src.ndim == 1:
            return self.rot.dot(vec_src) + self.trans
        elif vec_src.ndim == 2:
            return self.rot.dot(vec_src.T).T + self.trans
        else:
            assert False

    def inverse(self) -> "CoordinateTransform":
        rot_new = self.rot.T
        trans_new = -rot_new.dot(self.trans)
        return CoordinateTransform(trans_new, rot_new, self.dest, self.src)

    @classmethod
    def from_skrobot_coords(
        cls, coords: Coordinates, src: Optional[str] = None, dest: Optional[str] = None
    ):
        return cls(coords.worldpos(), coords.worldrot(), src, dest)

    def to_skrobot_coords(self) -> Coordinates:
        return Coordinates(self.trans, self.rot)


def chain_transform(
    tf_a2b: CoordinateTransform, tf_b2c: CoordinateTransform
) -> CoordinateTransform:
    if tf_a2b.dest is not None and tf_b2c.src is not None:
        assert tf_a2b.dest == tf_b2c.src, "{} does not match {}".format(tf_a2b.dest, tf_b2c.src)

    trans_a2c = tf_b2c.trans + tf_b2c.rot.dot(tf_a2b.trans)
    rot_a2c = tf_b2c.rot.dot(tf_a2b.rot)

    src_new = tf_a2b.src
    dest_new = tf_b2c.dest
    return CoordinateTransform(trans_a2c, rot_a2c, src_new, dest_new)


def test_transform():
    co_gripper2world = Coordinates([1, 1, 0])
    co_object2world = Coordinates([2.0, 0, 0])
    co_object2world.rotate(np.pi / 4, "z")
    tf_gripper2world = CoordinateTransform.from_skrobot_coords(co_gripper2world, "gripper", "world")
    tf_object2world = CoordinateTransform.from_skrobot_coords(co_object2world, "object", "world")
    tf_gripper2object = chain_transform(tf_gripper2world, tf_object2world.inverse())
    print(tf_gripper2object.to_skrobot_coords())


if __name__ == "__main__":
    test_transform()
