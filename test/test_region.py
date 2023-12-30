from typing import List, Tuple

import numpy as np
import pytest

from frmax2.metric import Metric
from frmax2.region import SuperlevelSet


def is_inside_sphere(x: np.ndarray) -> bool:
    return bool(np.linalg.norm(x) < 1.0)


@pytest.fixture(scope="session")
def sphere_dataset() -> Tuple[List[np.ndarray], List[bool]]:
    x_list = []
    y_list = []
    n_sample = 10000
    for _ in range(n_sample):
        x = np.random.uniform(-1.5, 1.5, 3)
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


def test_SuperlevelSet_itp(sphere_dataset: Tuple[List[np.ndarray], List[bool]]):
    metric = Metric.from_ls(np.ones(3))
    X, Y = sphere_dataset
    levelset = SuperlevelSet.fit(X, Y, metric)

    # slice into 1d
    param_slice = np.array([0.0, 0.0])
    itp = levelset.create_sliced_itp_object(param_slice, [0, 1], 100)
    values = itp(np.linspace(-2.0, 2.0, 100))
    positive_rate = np.sum(values > 0.0) / 100
    assert abs(positive_rate - 0.5) < 0.1

    param_slice = np.array([0.5, 0.0])
    itp = levelset.create_sliced_itp_object(param_slice, [0, 1], 100)
    positive_rate = np.sum(values > 0.0) / 100
    assert abs(positive_rate - np.sqrt(3) / 4.0) < 0.1
    assert np.isinf(itp(np.array([10.0]))[0])

    # slice into 2d
    param_slice = np.array([0.0])
    itp = levelset.create_sliced_itp_object(param_slice, [0], 100)
    X, Y = np.meshgrid(np.linspace(-2.0, 2.0, 100), np.linspace(-2.0, 2.0, 100))
    points = np.array([X.flatten(), Y.flatten()]).T
    values = itp(points)
    positive_rate = np.sum(values > 0.0) / 100**2
    assert abs(positive_rate - np.pi / 16.0) < 0.1

    param_slice = np.array([0.5])
    itp = levelset.create_sliced_itp_object(param_slice, [0], 100)
    values = itp(points)
    positive_rate = np.sum(values > 0.0) / 100**2
    assert abs(positive_rate - (np.sqrt(3) / 2) ** 2 * np.pi / 16.0) < 0.1


if __name__ == "__main__":
    test_SuperlevelSet(sphere_dataset())
