from typing import List, Tuple

import numpy as np
import pytest

from frmax2.metric import Metric
from frmax2.region import SuperlevelSet


def is_inside_sphere(x: np.ndarray) -> bool:
    return bool(np.linalg.norm(x) < 1.0)


def is_inside_ellipsoid(x: np.ndarray) -> bool:
    r_x, r_y, r_z = 1.0, 1.0, 0.5
    return bool(x[0] ** 2 / r_x**2 + x[1] ** 2 / r_y**2 + x[2] ** 2 / r_z**2 < 1.0)


@pytest.fixture(scope="session")
def sphere_dataset() -> Tuple[List[np.ndarray], List[bool]]:
    x_list = []
    y_list = []
    n_sample = 3000
    for _ in range(n_sample):
        x = np.random.uniform(-1.5, 1.5, 3)
        x_list.append(x)
        y = is_inside_sphere(x)
        y_list.append(y)
    return x_list, y_list


@pytest.fixture(scope="session")
def ellipsoid_dataset() -> Tuple[List[np.ndarray], List[bool]]:
    x_list = []
    y_list = []
    n_sample = 3000
    for _ in range(n_sample):
        x = np.random.uniform(-1.5, 1.5, 3)
        x_list.append(x)
        y = is_inside_ellipsoid(x)
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

    # check surface 1d
    surface = levelset.get_surface_by_slicing(np.zeros(2), [0, 1], 100)
    assert surface is not None
    assert len(surface.vertices) == 2
    # dist = np.linalg.norm(surface.vertices[0] - surface.vertices[1])
    volume = surface.volume()
    np.testing.assert_almost_equal(volume, 2.0, decimal=1.0)

    # check surface 2d
    surface = levelset.get_surface_by_slicing(np.zeros(1), [0], 200)
    vol = surface.volume()
    np.testing.assert_almost_equal(vol, 3.14159, decimal=1.0)


def test_SuperlevelSet_itp():
    for co_dim in range(1, 5):
        print(f"co_dim: {co_dim}")
        dim = co_dim + 1
        metric = Metric.from_ls(np.ones(dim))
        X = np.random.randn(10, dim)
        Y = np.random.randint(0, 2, 10).astype(bool)
        levelset = SuperlevelSet.fit(X, Y, metric)
        slice_point = np.array([0.1])
        itp = levelset.create_sliced_itp_object(slice_point, [0], 10, method="cubic")

        n_test = 1000
        X_test_subspace = np.random.randn(n_test, co_dim)
        X_test = np.hstack([np.tile(slice_point, (n_test, 1)), X_test_subspace])

        itp_values = itp(X_test_subspace)
        indices_finite = np.isfinite(itp_values)
        Y_approx = itp_values[indices_finite] > 0.0
        Y_gt = levelset.func(X_test)[indices_finite] > 0.0
        miss_classification = np.sum(Y_gt != Y_approx)
        rate = miss_classification / n_test
        print(f"rate: {rate}")
        assert rate < 0.05


if __name__ == "__main__":
    # test_SuperlevelSet(sphere_dataset())
    test_SuperlevelSet_itp()
