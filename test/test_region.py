from typing import List, Tuple

import numpy as np
import pytest

from frmax2.metric import Metric
from frmax2.region import SuperlevelSet


def is_inside_sphere(x: np.ndarray) -> bool:
    return np.linalg.norm(x) < 1.0


@pytest.fixture(scope="session")
def sphere_dataset() -> Tuple[List[np.ndarray], List[bool]]:
    x_list = []
    y_list = []
    n_sample = 10000
    for _ in range(n_sample):
        x = np.random.randn(3)
        x_list.append(x)
        y = is_inside_sphere(x)
        y_list.append(y)
    return x_list, y_list


def test_SuperlevelSet(sphere_dataset: Tuple[List[np.ndarray], List[bool]]):
    metric = Metric.from_ls(np.ones(3))
    X, Y = sphere_dataset
    levelset = SuperlevelSet.fit(X, Y, metric)

    # check miss classification is small
    miss_classification = 0
    for _ in range(1000):
        x = np.random.randn(3)
        if levelset.is_inside(x) != is_inside_sphere(x):
            miss_classification += 1
    assert miss_classification < 50

    # check grid points
    pts = levelset.create_grid_points(None, [], 10)
    assert len(pts) == 10**3

    pts = levelset.create_grid_points(np.array([0.0]), [0], 10)
    assert len(pts) == 10**2

    pts = levelset.create_grid_points(np.array([0.0, 1.0]), [0, 2], 10)
    assert len(pts) == 10**1
    assert np.linalg.norm(pts[:, 0] - 0.0) < 1e-4
    assert np.linalg.norm(pts[:, 2] - 1.0) < 1e-4

    # check surface
    surface = levelset.get_surface_by_slicing(np.zeros(2), [0, 1], 100)
    assert surface is not None
    assert len(surface.vertices) == 2
    dist = np.linalg.norm(surface.vertices[0] - surface.vertices[1])
    np.testing.assert_almost_equal(dist, 2.0, decimal=2)


if __name__ == "__main__":
    test_SuperlevelSet(sphere_dataset())
