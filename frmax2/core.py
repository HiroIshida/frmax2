import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from frmax2.metric import CompositeMetric, Metric
from frmax2.region import SuperlevelSet, get_co_axes

logger = logging.getLogger(__name__)


@dataclass
class ActiveSamplerConfig:
    n_mc_param_search: int = 300
    r_exploration: float = 0.5
    n_grid: int = 20
    aabb_margin: float = 0.5
    n_mc_integral: int = 100
    c_svm: float = 1e4


class SamplerCache:
    best_param_history: List[np.ndarray]
    best_volume_history: List[float]

    def __init__(self):
        self.best_param_history = []
        self.best_volume_history = []


class ActiveSamplerBase(ABC):
    fslset: SuperlevelSet
    metric: CompositeMetric
    is_valid_param: Callable[[np.ndarray], bool]
    config: ActiveSamplerConfig
    best_param_so_far: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    axes_param: List[int]
    sampler_cache: SamplerCache

    def __init__(
        self,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: CompositeMetric,
        param_init: np.ndarray,
        config: ActiveSamplerConfig = ActiveSamplerConfig(),
        is_valid_param: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        slset = SuperlevelSet.fit(X, Y, metric, C=config.c_svm)
        self.fslset = slset
        self.metric = metric
        self.config = config
        self.best_param_so_far = param_init
        if is_valid_param is None:
            self.is_valid_param = lambda x: True
        self.X = X
        self.Y = Y
        self.axes_param = list(range(len(param_init)))
        self.sampler_cache = SamplerCache()

    @abstractmethod
    def compute_sliced_volume(self, param: np.ndarray) -> float:
        ...

    @abstractmethod
    def sample_sliced_points(self, param: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def compute_sliced_widths(self, param: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update_metric(self) -> None:
        ...

    @property
    def dim(self) -> int:
        return self.metric.dim

    def _determine_param_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (param_candidates, volumes)"""

        def sample_until_valid(r_param: float) -> np.ndarray:
            while True:
                rand_params = param_metric.generate_random_inball(
                    param_center, self.config.n_mc_param_search, r_param
                )
                filtered = list(filter(self.is_valid_param, rand_params))
                if len(filtered) > 0:
                    return np.array(filtered)
                logger.debug("sample from ball again as no samples satisfies constraint")

        param_metric = self.metric.metirics[0]
        param_center = self.best_param_so_far
        logger.debug(f"current best param: {param_center}")
        logger.info(f"current best volume: {self.compute_sliced_volume(param_center)}")

        self.sampler_cache.best_param_history.append(param_metric)
        self.sampler_cache.best_volume_history.append(self.compute_sliced_volume(param_center))

        trial_count = 0
        r = self.config.r_exploration
        while True:
            param_sampled = sample_until_valid(r)
            volumes = np.array([self.compute_sliced_volume(p) for p in param_sampled])
            mean = np.mean(volumes)
            indices_better = np.where(volumes >= mean)[0]
            is_all_equal = len(indices_better) == self.config.n_mc_param_search
            if not is_all_equal:
                return param_sampled[indices_better], volumes[indices_better]
            trial_count += 1
            if trial_count % 10 == 9:
                r *= 1.5  # increase radius to explore more
        assert False

    def ask(self) -> np.ndarray:
        param_cands, volumes = self._determine_param_candidates()
        # update param_best_so_far
        idx_max_volume = np.argmax(volumes)
        if self.compute_sliced_volume(self.best_param_so_far) < volumes[idx_max_volume]:
            self.best_param_so_far = param_cands[idx_max_volume]

        # sample points
        x_best = None
        uncertainty_max = -np.inf
        for param in param_cands:
            co_points = self.sample_sliced_points(param)
            for co_point in co_points:
                x = np.hstack([param, co_point])
                uncertainty = np.min(self.metric(x, self.X))
                if uncertainty > uncertainty_max:
                    uncertainty_max = uncertainty
                    x_best = x
        assert x_best is not None
        return x_best

    def tell(self, x: np.ndarray, y: bool) -> None:
        self.update_metric()
        X = np.vstack([self.X, x])
        Y = np.hstack([self.Y, y])
        self.X = X
        self.Y = Y
        self.fslset = SuperlevelSet.fit(X, Y, self.metric, C=self.config.c_svm)


class HolllessActiveSampler(ActiveSamplerBase):
    def update_metric(self) -> None:
        logger.debug("update non-parameter-side metric")

        ls_co = np.sqrt(np.diag(self.metric.metirics[1].cmat))
        logger.debug(f"current ls_co: {ls_co}")

        param_pre = self.X[-1][self.axes_param]
        width = self.compute_sliced_widths(param_pre)
        ls_co_cand = width * 0.25
        logger.debug(f"current ls_co_cand: {ls_co_cand}")

        r = 1.5
        ls_co_min = ls_co * (1 / r)
        ls_co_max = ls_co * r
        logger.debug(f"ls_co_min: {ls_co_min}, ls_co_max: {ls_co_max}")
        ls_co = np.max(np.vstack((ls_co_cand, ls_co_min)), axis=0)
        ls_co = np.min(np.vstack((ls_co, ls_co_max)), axis=0)

        logger.debug(f"determined ls_co: {ls_co}")

        metric_co = Metric.from_ls(ls_co)
        new_metric = CompositeMetric([self.metric.metirics[0], metric_co])
        self.metric = new_metric

    def compute_sliced_volume(self, param: np.ndarray) -> float:
        return self.fslset.sliced_volume_grid(param, self.axes_param, self.config.n_grid)

    def sample_sliced_points(self, param: np.ndarray) -> np.ndarray:
        surface = self.fslset.get_surface_by_slicing(param, self.axes_param, self.config.n_grid)
        if surface is None:
            return np.zeros((0, self.metric.metirics[0].dim))
        else:
            return surface.points

    def compute_sliced_widths(self, param: np.ndarray) -> np.ndarray:
        return self.fslset.measure_region_widths_grid(param, self.axes_param, self.config.n_grid)


class NaiveActiveSampler(ActiveSamplerBase):
    def update_metric(self) -> None:
        pass

    def compute_sliced_volume(self, param: np.ndarray) -> float:
        return self.fslset.sliced_volume_mc(param, self.axes_param, self.config.n_mc_integral)

    def sample_sliced_points(self, param: np.ndarray) -> np.ndarray:
        points = self.fslset.sample_mc_points_sliced(
            param, self.axes_param, self.config.n_mc_integral
        )
        points_inside = points[self.fslset.func(points) > -0.0]
        axes_co = get_co_axes(self.dim, self.axes_param)
        return points_inside[:, axes_co]

    def compute_sliced_widths(self, param: np.ndarray) -> np.ndarray:
        return self.fslset.measure_region_widths_mc(
            param, self.axes_param, self.config.n_mc_integral
        )
