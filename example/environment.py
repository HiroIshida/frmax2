import math
from abc import ABC
from typing import Tuple

import numpy as np


def npdf(dist_from_center: float) -> float:
    return math.exp(-0.5 * dist_from_center**2)


class GaussianEnvironment(ABC):
    def __init__(self, n_dim: int, m_dim: int):
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.name = "gaussian"

    def _npdf(self, param):
        dists = np.sqrt(np.sum(param**2))
        return npdf(dists)

    def evaluate_size(self, param: np.ndarray) -> float:
        assert param.ndim == 1, "must be 1"
        f = self._npdf(param)

        if self.m_dim == 1:
            return (2 * f).item()  # adhoc item
        elif self.m_dim == 2:
            return math.pi * f**2
        elif self.m_dim == 3:
            return 4 * math.pi * f**3 / 3.0

    def isInside(self, x: np.ndarray) -> bool:
        assert x.ndim == 1, "must be 1"
        return np.linalg.norm(x[-self.m_dim :]) < self._npdf(x[0 : self.n_dim])

    def default_init_param(self) -> np.ndarray:
        return np.array([-2.0] + [0 for i in range(self.n_dim - 1)])

    def default_lengthscales(self) -> Tuple[np.ndarray, np.ndarray]:
        ls_fr = np.array([1.0] * self.m_dim)
        ls_param = np.array([0.5] * self.n_dim) * 0.5
        return ls_param, ls_fr

    def visualize_region(self, param_min, param_max, fax) -> None:
        param_lin = np.linspace(param_min, param_max, 200)
        upper = np.exp(-0.5 * param_lin**2)
        lower = -upper
        fig, ax = fax
        ax.fill_between(param_lin, lower, upper, color="black", alpha=0.15, edgecolor="white")
