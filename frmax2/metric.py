from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
from scipy.linalg import block_diag, sqrtm
from sklearn.metrics import pairwise


class MetricBase(ABC):
    @property
    @abstractmethod
    def cmat(self) -> np.ndarray:
        ...

    @cached_property
    def tensor(self):
        return np.linalg.inv(self.cmat)

    @property
    def sqrt_tensor(self):
        return sqrtm(np.linalg.inv(self.cmat))

    def kern(self, X_, Y_=None):
        if Y_ is None:
            Y_ = X_
        X, Y = X_.dot(self.sqrt_tensor), Y_.dot(self.sqrt_tensor)
        res = pairwise.rbf_kernel(X, Y)
        return res

    def __call__(self, x, X):
        if X.ndim == 1:
            n_point = 1
        else:
            n_point = X.shape[0]
        diffs = X - np.tile(x, (n_point, 1))
        tmp = diffs.dot(self.tensor)
        dists = np.sqrt(np.sum(tmp * diffs, axis=1))
        return dists

    def generate_random_inball(self, center, N, radius=1.0):
        """
        Generating a random sample uniformely inside a high dimensional ball is done by
        Barthe, Franck, et al. "A probabilistic approach to the geometry of the $l_{p}^n$-ball." The Annals of Probability 33.2 (2005): 480-513.

        http://mathworld.wolfram.com/BallPointPicking.html
        is wrong. lambda must be 0.5, which means we must set radius in numpy.random.exponetial to be 2.0
        """
        dim = len(center)
        y = np.random.exponential(scale=2, size=(N))
        X = np.random.randn(dim, N)
        denom = np.sqrt(np.sum(X**2, axis=0) + y)
        rands_inball = X / np.tile(denom, (dim, 1))

        L_inv = np.linalg.inv(self.sqrt_tensor)
        rands_inellipsoid = L_inv.dot(rands_inball) * radius
        return rands_inellipsoid.T + np.tile(center, (N, 1))


@dataclass
class Metric(MetricBase):
    _cmat: np.ndarray

    @property
    def cmat(self) -> np.ndarray:
        return self._cmat

    @classmethod
    def from_ls(cls, ls: np.ndarray):
        return cls(np.diag(ls**2))

    @property
    def dim(self) -> int:
        return self.cmat.shape[0]


@dataclass
class CompositeMetric(MetricBase):
    metirics: List[Metric]

    @classmethod
    def from_ls_list(cls, ls_list: List[np.ndarray]):
        return cls([Metric.from_ls(ls) for ls in ls_list])

    @property
    def cmat(self) -> np.ndarray:
        return block_diag(*[m.cmat for m in self.metirics])

    @property
    def dim(self) -> int:
        return sum([m.dim for m in self.metirics])
