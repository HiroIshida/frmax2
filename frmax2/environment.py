import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.patches import Circle
from scipy.special import gamma


def npdf(dist_from_center: float) -> float:
    return math.exp(-0.5 * dist_from_center**2)


def generate_random_orthogonal_matrix(size):
    random_matrix = np.random.rand(size, size)
    Q, R = np.linalg.qr(random_matrix)
    return Q


class EnvironmentBase(ABC):
    def __init__(
        self,
        n_dim: int,
        m_dim: int,
        with_bias: bool = False,
        with_hollow: bool = False,
        error_consider_axes: Optional[List[int]] = None,
        random_basis: bool = False,
    ):
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
        if error_consider_axes is None:
            error_consider_axes = list(range(self.m_dim))
        self.error_consider_axes = np.array(error_consider_axes, dtype=int)
        if random_basis:
            self.M = generate_random_orthogonal_matrix(self.n_dim)
        else:
            self.M = np.eye(self.n_dim)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones(self.m_dim) * -1.5, np.ones(self.m_dim) * 1.5

    @property
    def sampling_space_volume(self) -> np.ndarray:
        return np.prod(self.bounds[1] - self.bounds[0])

    def sample_situation(self) -> np.ndarray:
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def radius_func(self, param: np.ndarray) -> float:
        param = np.dot(self.M, param)
        return self._radius_func(param)

    @abstractmethod
    def _radius_func(self, param: np.ndarray) -> float:
        pass

    def evaluate_size(self, param: np.ndarray) -> float:
        R = self.radius_func(param)
        r = self.hollow_scale * R
        m_consider = len(self.error_consider_axes)
        compute_volume = (
            lambda r: np.pi ** (m_consider / 2) / (gamma(m_consider / 2 + 1)) * r**m_consider
        )
        R_volume = compute_volume(R)
        if r > 1e-6:
            r_volume = compute_volume(r)
        else:
            r_volume = 0.0

        # because ignored dimension is filled with feasible region so we must multiply it
        axes_ignored = np.array(list(set(list(range(self.m_dim))) - set(self.error_consider_axes)))
        if len(axes_ignored) == 0:
            volume_ignored = 1.0
        else:
            volume_ignored = np.prod(self.bounds[1][axes_ignored] - self.bounds[0][axes_ignored])
        return (R_volume - r_volume) * volume_ignored

    def isInside(self, x: np.ndarray) -> bool:
        assert x.ndim == 1, "must be 1"
        e, theta = x[-self.m_dim :], x[0 : self.n_dim]
        e_consider = e[self.error_consider_axes]
        f_value = self.radius_func(theta)
        bias = self.bias_param * f_value
        inside_outer = bool(np.linalg.norm(e_consider - bias) < f_value)
        inside_inner = bool(np.linalg.norm(e_consider - bias) < self.hollow_scale * f_value)
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


class GaussianEnvironment(EnvironmentBase):
    def _radius_func(self, param: np.ndarray) -> float:
        dists = np.sqrt(np.sum(param**2))
        return npdf(dists)


class AnisoEnvironment(EnvironmentBase):
    def _radius_func(self, param: np.ndarray) -> float:
        weight = np.ones(self.n_dim) * 0.05
        weight[0] = 1.0
        weight[1] = 0.5
        param_ = param * weight
        dists = np.sqrt(np.sum(param_**2))
        return npdf(dists)
