from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional

import numpy as np
import trimesh
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.primitives import Box, MeshLink

import pybullet


class PybulletPrimitiveMixIn:
    id_value: int
    obj: Link

    def set_coords(self, co: Coordinates):
        pybullet.resetBasePositionAndOrientation(self.id_value, co.worldpos(), co.quaternion)
        self.obj.newcoords(co)


class PybulletBox(PybulletPrimitiveMixIn):
    id_value: int
    obj: Box

    def __init__(
        self,
        extents: np.ndarray,
        pos: Optional[np.ndarray] = None,
        rot: Optional[np.ndarray] = None,
        with_sdf: bool = True,
    ):
        vis_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=0.5 * extents)
        col_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=0.5 * extents)
        self.id_value = pybullet.createMultiBody(
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
        )
        box = Box(extents=extents, with_sdf=with_sdf)
        self.obj = box
        co = Coordinates(pos=pos, rot=rot)
        self.set_coords(co)

    @property
    def sdf(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        return self.obj.sdf


class PybulletMesh(PybulletPrimitiveMixIn):
    id_value: int
    obj: MeshLink

    def __init__(
        self,
        mesh_path: Path,
        scale: float = 1.0,
        pos: Optional[np.ndarray] = None,
        rot: Optional[np.ndarray] = None,
        with_sdf: bool = False,
    ):
        self.id_value = self.load_mesh(mesh_path, scale=scale)

        mesh = trimesh.load_mesh(str(mesh_path.expanduser()))
        matrix = np.eye(4)
        matrix[:2, :2] *= scale
        mesh.apply_transform(matrix)
        self.obj = MeshLink(visual_mesh=mesh, with_sdf=with_sdf)

        co = Coordinates(pos=pos, rot=rot)
        self.set_coords(co)

    @staticmethod
    def load_mesh(mesh_path: Path, scale: float = 1.0) -> int:
        tmp_urdf_file = """
            <?xml version="1.0" ?>
            <robot name="tmp">
            <link name="base_link" concave="yes">
            <visual>
            <geometry>
                <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
            </geometry>
            <material name="">
              <color rgba="0.6 0.6 0.6 1.0" />
            </material>
            </visual>
            <collision concave="yes">
            <geometry>
            <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
            </geometry>
            </collision>
            </link>
            </robot>
        """.format(
            mesh_path=str(mesh_path), scale=scale
        )
        # save urdf file to temporary file
        with TemporaryDirectory() as td:
            urdf_file_path = Path(td) / "tmp.urdf"
            with open(urdf_file_path, "w") as f:
                f.write(tmp_urdf_file)
            obj_id = pybullet.loadURDF(str(urdf_file_path))
        return obj_id
