import math
from typing import Tuple

import numpy as np
from matplotlib.patches import Circle


def npdf(dist_from_center: float) -> float:
    return math.exp(-0.5 * dist_from_center**2)


class GaussianEnvironment:
    def __init__(self, n_dim: int, m_dim: int, with_bias: bool = False, with_hollow: bool = False):
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.name = "gaussian"
        if with_bias:
            self.bias_param = 0.8
        else:
            self.bias_param = 0.0
        if with_hollow:
            self.hollow_scale = 0.2
        else:
            self.hollow_scale = 0.0
        self.with_hollow = with_hollow
        self.with_bias = with_bias

    def _npdf(self, param):
        dists = np.sqrt(np.sum(param**2))
        return npdf(dists)

    def evaluate_size(self, param: np.ndarray) -> float:
        assert param.ndim == 1, "must be 1"
        f = self._npdf(param)

        if self.m_dim == 1:
            return 2 * f * (1.0 - self.hollow_scale)
        elif self.m_dim == 2:
            assert not self.with_hollow
            return math.pi * f**2 * (1 - self.hollow_scale**2)
        elif self.m_dim == 3:
            assert not self.with_hollow
            return 4 * math.pi * f**3 * (1 - iself.hollow_scale**3) / 3.0

    def isInside(self, x: np.ndarray) -> bool:
        assert x.ndim == 1, "must be 1"
        e, theta = x[-self.m_dim :], x[0 : self.n_dim]
        f_value = self._npdf(theta)
        bias = self.bias_param * f_value
        inside_outer = bool(np.linalg.norm(e - bias) < f_value)
        inside_inner = bool(np.linalg.norm(e - bias) < self.hollow_scale * f_value)
        return inside_outer and not inside_inner

    def default_init_param(self) -> np.ndarray:
        return np.array([-2.0] + [0 for i in range(self.n_dim - 1)])

    def default_lengthscales(self) -> Tuple[np.ndarray, np.ndarray]:
        ls_fr = np.array([1.0] * self.m_dim)
        ls_param = np.array([0.5] * self.n_dim) * 0.5
        return ls_param, ls_fr

    def visualize_region(self, param_min, param_max, fax) -> None:
        param_lin = np.linspace(param_min, param_max, 200)
        f_value = np.exp(-0.5 * param_lin**2)
        bias = self.bias_param * f_value
        upper_outer = f_value + bias
        lower_outer = -f_value + bias
        upper_inner = self.hollow_scale * f_value + bias
        lower_inner = -self.hollow_scale * f_value + bias
        fig, ax = fax
        if self.with_hollow:
            ax.fill_between(
                param_lin, upper_inner, upper_outer, color="black", alpha=0.15, edgecolor="white"
            )
            ax.fill_between(
                param_lin, lower_outer, lower_inner, color="black", alpha=0.15, edgecolor="white"
            )
        else:
            ax.fill_between(
                param_lin, lower_outer, upper_outer, color="black", alpha=0.15, edgecolor="white"
            )

    def visualize_optimal_sliced_region(self, fax) -> None:
        fig, ax = fax
        assert self.m_dim == 2
        circle = Circle((self.bias_param, self.bias_param), 1.0, fill=False, color="red")
        ax.add_patch(circle)
