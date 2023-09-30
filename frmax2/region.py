from dataclasses import dataclass
from math import factorial
from typing import Callable, List, Optional

import numpy as np
from skimage import measure
from sklearn.svm import SVC

from frmax2.metric import CompositeMetric, MetricBase
from frmax2.utils import get_co_axes


def rescale(pts, b_min, b_max, n):
    n_points, n_dim = pts.shape
    width = b_max - b_min
    b_min_tile = np.tile(b_min, (n_points, 1))
    width_tile = np.tile(width, (n_points, 1))
    pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
    return pts_rescaled


def outer_surface_idx(surface_list):
    def bounding_box_size(surf):
        b_max = np.max(surf, axis=0)
        b_min = np.min(surf, axis=0)
        width = b_max - b_min
        boxsize = np.prod(width)
        return boxsize

    # circumscribe outer box
    sizes = [bounding_box_size(surf) for surf in surface_list]
    idx = np.argmax(sizes)
    return idx


@dataclass(frozen=True)
class Surface:
    vertices: np.ndarray
    edges: np.ndarray

    def __post_init__(self):
        for e in [self.vertices, self.edges]:
            e.flags.writeable = False

    @property
    def ambient_dim(self) -> int:
        return len(self.vertices[0])

    def volume(self) -> float:
        if self.ambient_dim == 1:
            return abs(self.vertices[1] - self.vertices[0])

        # use Stokes's theorem
        vol = 0.0
        for e in self.edges:
            facet_matrix = self.vertices[e, :]
            vol += np.linalg.det(facet_matrix)
        return abs(vol) / factorial(self.ambient_dim)

    @property
    def points(self) -> np.ndarray:
        return self.vertices


@dataclass(frozen=True)
class SuperlevelSet:
    b_min: np.ndarray
    b_max: np.ndarray
    func: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self):
        for e in [self.b_min, self.b_max]:
            e.flags.writeable = False

    @property
    def dim(self) -> int:
        return len(self.b_min)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: MetricBase,
        C: float = 1e8,
        margin: float = 0.5,
    ) -> "SuperlevelSet":

        kernel = metric.gen_aniso_rbf_kernel()
        svc = SVC(gamma="auto", kernel=kernel, probability=False, C=C)

        # check both positive and negative samples exist
        X_: np.ndarray = np.array(X, dtype=float)
        Y_: np.ndarray = np.array(Y, dtype=bool)
        assert np.any(Y_) and np.any(~Y_)
        svc.fit(X_, Y_)
        b_min_tmp, b_max_tmp = X_.min(axis=0), X_.max(axis=0)
        width_tmp = b_max_tmp - b_min_tmp
        b_min = b_min_tmp - width_tmp * margin
        b_max = b_max_tmp + width_tmp * margin
        return cls(b_min, b_max, svc.decision_function)

    def create_grid_points(
        self, point_slice: Optional[np.ndarray], axes_slice: List[int], n_grid: int
    ) -> np.ndarray:

        if point_slice is None:
            assert len(axes_slice) == 0
        else:
            assert len(point_slice) == len(axes_slice)

        # NOTE: co indicates a value is about complement axes
        axes_co = get_co_axes(self.dim, axes_slice)
        b_min_co = self.b_min[axes_co]
        b_max_co = self.b_max[axes_co]
        linspace_comp_list = [
            np.linspace(b_min_co[i], b_max_co[i], n_grid) for i in range(len(axes_co))
        ]
        meshgrid_comp_list = np.meshgrid(*linspace_comp_list)
        meshgrid_comp_flatten_list = [m.flatten() for m in meshgrid_comp_list]
        grid_points_comp = np.array(list(zip(*meshgrid_comp_flatten_list)))

        grid_points = np.zeros((len(grid_points_comp), self.dim))
        grid_points[:, axes_co] = grid_points_comp
        if point_slice is not None:
            grid_points[:, axes_slice] = point_slice
        return grid_points

    def is_inside(self, x: np.ndarray) -> bool:
        bools = self.func(np.expand_dims(x, axis=0)) > 0
        return bool(bools[0])

    def get_surface_by_slicing(
        self, point_slice: Optional[np.ndarray], axes_slice: List[int], n_grid: int
    ) -> Optional[Surface]:
        grid_points = self.create_grid_points(point_slice, axes_slice, n_grid)
        values = self.func(grid_points)

        dim_co = self.dim - len(axes_slice)
        axes_co = get_co_axes(self.dim, axes_slice)
        b_min_co = self.b_min[axes_co]
        b_max_co = self.b_max[axes_co]

        if dim_co == 1:
            return self._get_surface_by_slicing_1d(values, b_min_co, b_max_co, n_grid)
        elif dim_co == 2:
            return self._get_surface_by_slicing_2d(values, b_min_co, b_max_co, n_grid)
        else:
            assert False, "under constrction"

    @staticmethod
    def _get_surface_by_slicing_1d(
        values: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, n_grid: int
    ) -> Optional[Surface]:

        if 0 in values:
            raise ValueError("we assume values don't have 0 inside")
        values_left = values[0:-1]
        values_right = values[1:]
        mul = values_left * values_right
        idxes_surface = np.where((mul < 0))[0]

        b_min_ = b_min.item()
        b_max.item()
        points = np.linspace(b_min_, b_max, n_grid)

        simplexes = []
        for idx in idxes_surface:
            value_left = abs(values[idx])
            value_right = abs(values[idx + 1])
            point = (value_right * points[idx] + value_left * points[idx + 1]) / (
                value_left + value_right
            )
            simplexes.append(point)

        if len(simplexes) == 0:
            return None

        if len(simplexes) > 2:
            points = np.vstack([simplexes[0], simplexes[-1]])
        else:
            points = np.vstack(simplexes)
        if len(points) != 2:
            # I don't know why this happens...
            # at least in the original implementation, this never happens
            return None
        return Surface(points, np.array([0, 1], dtype=int))

    @staticmethod
    def _get_surface_by_slicing_2d(
        values: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, n_grid: int
    ) -> Optional[Surface]:

        data = values.reshape(n_grid, n_grid).T
        contours_ = measure.find_contours(data, 0.0)

        contours = [rescale(pts, b_min, b_max, n_grid) for pts in contours_]
        contours_closed = list(filter(lambda cn: np.all(cn[0, :] == cn[-1, :]), contours))

        if len(contours_closed) == 1:
            vertices = contours_closed[0]
        elif len(contours_closed) > 1:
            idx_outer = outer_surface_idx(contours_closed)
            vertices = contours_closed[idx_outer]
        else:
            return None
        n_vert = len(vertices)
        edges = np.column_stack((np.arange(n_vert), np.arange(1, n_vert + 1) % n_vert))
        return Surface(vertices, edges)


@dataclass(frozen=True)
class FactorizableSuperLevelSet:
    slset: SuperlevelSet
    metric: CompositeMetric
    b_min: np.ndarray
    b_max: np.ndarray
    n_grid: int
    margin: float
    C: float
    X: np.ndarray
    Y: np.ndarray

    @classmethod
    def fit(
        cls,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: CompositeMetric,
        n_grid: int,
        C: float = 1e8,
        margin: float = 0.5,
    ) -> "FactorizableSuperLevelSet":
        slset = SuperlevelSet.fit(X, Y, metric, C, margin)
        b_min = np.min(X, axis=0)
        b_max = np.max(X, axis=0)
        w = b_max - b_min
        b_min -= w * margin
        b_max += w * margin
        return cls(slset, metric, b_min, b_max, n_grid, margin, C, X, Y)

    @property
    def dim(self) -> int:
        return self.metric.metirics[0].dim

    @property
    def axes_slice(self) -> List[int]:
        return list(range(self.metric.metirics[0].dim))

    def sample_points_sliced(self, point: np.ndarray) -> np.ndarray:
        surface = self.slset.get_surface_by_slicing(point, self.axes_slice, self.n_grid)
        if surface is None:
            return np.zeros((0, self.dim))
        else:
            return surface.points

    def volume_sliced(self, point: np.ndarray) -> float:
        surface = self.slset.get_surface_by_slicing(point, self.axes_slice, self.n_grid)
        if surface is None:
            return 0.0
        else:
            return surface.volume()

    def region_widths(self, point: np.ndarray) -> np.ndarray:
        surface = self.slset.get_surface_by_slicing(point, self.axes_slice, self.n_grid)
        if surface is None:
            return np.zeros(self.dim)
        get_co_axes(self.dim, self.axes_slice)
        co_points = surface.points
        b_min = np.min(co_points, axis=0)
        b_max = np.max(co_points, axis=0)
        return b_max - b_min
