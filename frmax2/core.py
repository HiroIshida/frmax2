import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

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
    c_svm_min: float = 10
    c_svm_reduction_rate: float = 1.0
    n_process: int = 1  # if > 1, use multiprocessing
    learning_rate: float = 1.0
    integration_method: Literal["mc", "grid"] = "grid"
    measure_width_method: Literal["mc", "grid"] = "grid"
    sample_error_method: Literal["mc-margin", "mc-inside", "grid"] = "grid"
    box_cut: bool = False


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
    count_additional: int
    c_svm_current: float

    def __init__(
        self,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: CompositeMetric,
        param_init: np.ndarray,
        config: ActiveSamplerConfig = ActiveSamplerConfig(),
        is_valid_param: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        c_svm_current = config.c_svm
        slset = SuperlevelSet.fit(X, Y, metric, C=c_svm_current, box_cut=config.box_cut)
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
        self.c_svm_current = c_svm_current
        self.count_additional = 0

    @staticmethod
    def _compute_sliced_volume_inner(
        fslset: SuperlevelSet, param: np.ndarray, config: ActiveSamplerConfig
    ) -> float:
        axes_param = list(range(len(param)))
        if config.integration_method == "mc":
            return fslset.sliced_volume_mc(param, axes_param, config.n_mc_integral)
        elif config.integration_method == "grid":
            return fslset.sliced_volume_grid(param, axes_param, config.n_grid)
        else:
            assert False

    def compute_sliced_volume(self, param: np.ndarray) -> float:
        return self._compute_sliced_volume_inner(self.fslset, param, self.config)

    def compute_sliced_volume_batch(self, params: List[np.ndarray]) -> np.ndarray:
        if self.config.n_process == 1:
            return np.array(
                [
                    self._compute_sliced_volume_inner(self.fslset, param, self.config)
                    for param in params
                ]
            )
        else:
            n_param = len(params)
            with ProcessPoolExecutor(self.config.n_process) as executor:
                results = executor.map(
                    self._compute_sliced_volume_inner,
                    [self.fslset] * n_param,
                    params,
                    [self.config] * n_param,
                )
            return np.array(list(results))

    def sample_sliced_points(self, param: np.ndarray) -> np.ndarray:
        axes_co = get_co_axes(self.dim, self.axes_param)
        if self.config.sample_error_method == "mc-margin":
            points = self.fslset.sample_mc_points_sliced(
                param, self.axes_param, self.config.n_mc_integral
            )
            points_inside = points[np.abs(self.fslset.func(points)) < 1.0]
            return points_inside[:, axes_co]
        elif self.config.sample_error_method == "mc-inside":
            points = self.fslset.sample_mc_points_sliced(
                param, self.axes_param, self.config.n_mc_integral
            )
            negative_side_f_value = -1.0
            points_inside = points[self.fslset.func(points) > negative_side_f_value]
            return points_inside[:, axes_co]
        elif self.config.sample_error_method == "grid":
            surface = self.fslset.get_surface_by_slicing(param, self.axes_param, self.config.n_grid)
            if surface is None:
                return np.zeros((0, self.metric.metirics[0].dim))
            else:
                return surface.points
        else:
            assert False

    def compute_sliced_widths(self, param: np.ndarray) -> np.ndarray:
        if self.config.measure_width_method == "mc":
            return self.fslset.measure_region_widths_mc(
                param, self.axes_param, self.config.n_mc_integral
            )
        elif self.config.measure_width_method == "grid":
            return self.fslset.measure_region_widths_grid(
                param, self.axes_param, self.config.n_grid
            )
        else:
            assert False

    @abstractmethod
    def update_metric(self) -> None:
        ...

    @abstractmethod
    def check_config_compat(self) -> bool:
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

        self.sampler_cache.best_param_history.append(param_center)
        self.sampler_cache.best_volume_history.append(self.compute_sliced_volume(param_center))

        trial_count = 0
        r = self.config.r_exploration
        while True:
            param_sampled = sample_until_valid(r)
            volumes = self.compute_sliced_volume_batch(list(param_sampled))
            mean = np.mean(volumes)
            indices_better = np.where(volumes >= mean)[0]
            is_all_equal = len(indices_better) == self.config.n_mc_param_search
            if not is_all_equal and len(indices_better) > 0:
                return param_sampled[indices_better], volumes[indices_better]
            trial_count += 1
            r *= 1.1  # increase radius to explore more
            print(f"trial count: {trial_count}")
            if trial_count == 10:
                assert False
        assert False

    def _ask(self) -> List[Tuple[np.ndarray, float]]:
        assert self.count_additional == 0
        param_cands, volumes = self._determine_param_candidates()
        assert len(param_cands) > 0
        # update param_best_so_far
        best_param = None
        idx_max_volume = np.argmax(volumes)
        if self.compute_sliced_volume(self.best_param_so_far) < volumes[idx_max_volume]:
            best_param = param_cands[idx_max_volume]
        if best_param is not None:
            self.best_param_so_far += self.config.learning_rate * (
                best_param - self.best_param_so_far
            )

        x_uncertainty_pairs = []
        for param in param_cands:
            co_points = self.sample_sliced_points(param)
            for co_point in co_points:
                x = np.hstack([param, co_point])
                assert len(x) == self.dim
                uncertainty = np.min(self.metric(x, self.X))
                x_uncertainty_pairs.append((x, uncertainty))
        return x_uncertainty_pairs

    def ask(self) -> np.ndarray:
        x_uncertainty_pairs = self._ask()
        x_uncertainty_pairs.sort(key=lambda x: -x[1])
        return x_uncertainty_pairs[0][0]

    def ask_n_best(self, n: int) -> List[np.ndarray]:
        x_uncertainty_pairs = self._ask()
        x_uncertainty_pairs.sort(key=lambda x: -x[1])
        x_best = x_uncertainty_pairs[0][0]
        x_uncertainty_pairs = x_uncertainty_pairs[1:]
        np.random.shuffle(x_uncertainty_pairs)
        if len(x_uncertainty_pairs) < n:
            return [x_best] + [x for x, _ in x_uncertainty_pairs]
        else:
            return [x_best] + [x for x, _ in x_uncertainty_pairs[: n - 1]]

    def ask_additional(self) -> np.ndarray:
        param_here = self.best_param_so_far
        sliced_points = self.sample_sliced_points(param_here)
        assert len(sliced_points) > 0
        logger.debug(f"slice points: {sliced_points}")
        if self.count_additional == 0:
            e_new = sliced_points[0]  # because we cannot compare
        else:
            e_dim = sliced_points.shape[1]
            E_additional_so_far = self.X[-self.count_additional :, -e_dim:]
            e_metric = self.metric.metirics[1]
            uncertainty_max = -np.inf
            e_new = None
            for e in sliced_points:
                uncertainty = np.min(e_metric(e, E_additional_so_far))
                if uncertainty > uncertainty_max:
                    uncertainty_max = uncertainty
                    e_new = e
            assert e_new is not None and len(e_new) == e_dim
        self.count_additional += 1
        return np.hstack([param_here, e_new])

    def tell(self, x: np.ndarray, y: bool) -> None:
        assert x.ndim == 1
        X = np.array([x])
        Y = np.array([y])
        self.tell_multi(X, Y)

    def tell_multi(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert X.ndim == 2
        self.c_svm_current = max(
            self.config.c_svm_min, self.c_svm_current * self.config.c_svm_reduction_rate
        )
        logger.debug(f"current c_svm: {self.c_svm_current}")
        self.update_metric()
        X = np.vstack([self.X, X])
        Y = np.hstack([self.Y, Y])
        self.X = X
        self.Y = Y
        self.fslset = SuperlevelSet.fit(
            X, Y, self.metric, C=self.config.c_svm, box_cut=self.config.box_cut
        )


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

    def check_config_compat(self) -> bool:
        return self.config.sample_error_method != "mc-inside"


class DistributionGuidedSampler:
    fslset: SuperlevelSet
    metric: CompositeMetric
    is_valid_param: Callable[[np.ndarray], bool]
    config: ActiveSamplerConfig
    best_param_so_far: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    axes_param: List[int]
    sampler_cache: SamplerCache
    count_additional: int
    c_svm_current: float
    situation_sampler: Callable[[], np.ndarray]

    def __init__(
        self,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: CompositeMetric,
        param_init: np.ndarray,
        situation_sampler: Callable[[], np.ndarray],
        config: ActiveSamplerConfig = ActiveSamplerConfig(),
        is_valid_param: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        c_svm_current = config.c_svm
        slset = SuperlevelSet.fit(X, Y, metric, C=c_svm_current, box_cut=config.box_cut)
        self.fslset = slset
        self.situation_sampler = situation_sampler
        self.metric = metric
        self.config = config
        self.best_param_so_far = param_init
        if is_valid_param is None:
            self.is_valid_param = lambda x: True
        self.X = X
        self.Y = Y
        self.axes_param = list(range(len(param_init)))
        self.sampler_cache = SamplerCache()
        self.c_svm_current = c_svm_current
        self.count_additional = 0

    @property
    def dim(self) -> int:
        # perfectly copied from ActiveSamplerBase
        return self.metric.dim

    def tell(self, x: np.ndarray, y: bool) -> None:
        # perfectly copied from ActiveSamplerBase
        assert x.ndim == 1
        X = np.array([x])
        Y = np.array([y])
        self.tell_multi(X, Y)

    def tell_multi(self, X: np.ndarray, Y: np.ndarray) -> None:
        # perfectly copied from ActiveSamplerBase
        assert X.ndim == 2
        self.c_svm_current = max(
            self.config.c_svm_min, self.c_svm_current * self.config.c_svm_reduction_rate
        )
        logger.debug(f"current c_svm: {self.c_svm_current}")
        X = np.vstack([self.X, X])
        Y = np.hstack([self.Y, Y])
        self.X = X
        self.Y = Y
        self.fslset = SuperlevelSet.fit(
            X, Y, self.metric, C=self.config.c_svm, box_cut=self.config.box_cut
        )

    def compute_sliced_volume(self, param: np.ndarray) -> float:
        # perfectly copied from ActiveSamplerBase
        return self._compute_sliced_volume_inner(self.fslset, param, self.config)

    def compute_sliced_volume_batch(self, params: List[np.ndarray]) -> np.ndarray:
        # perfectly copied from ActiveSamplerBase
        if self.config.n_process == 1:
            return np.array(
                [
                    self._compute_sliced_volume_inner(self.fslset, param, self.config)
                    for param in params
                ]
            )
        else:
            n_param = len(params)
            with ProcessPoolExecutor(self.config.n_process) as executor:
                results = executor.map(
                    self._compute_sliced_volume_inner,
                    [self.fslset] * n_param,
                    params,
                    [self.config] * n_param,
                )
            return np.array(list(results))

    def _determine_param_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        # perfectly copied from ActiveSamplerBase

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

        self.sampler_cache.best_param_history.append(param_center)
        self.sampler_cache.best_volume_history.append(self.compute_sliced_volume(param_center))

        trial_count = 0
        r = self.config.r_exploration
        while True:
            param_sampled = sample_until_valid(r)
            volumes = self.compute_sliced_volume_batch(list(param_sampled))
            mean = np.mean(volumes)
            indices_better = np.where(volumes >= mean)[0]
            is_all_equal = len(indices_better) == self.config.n_mc_param_search
            if not is_all_equal and len(indices_better) > 0:
                return param_sampled[indices_better], volumes[indices_better]
            trial_count += 1
            r *= 1.1  # increase radius to explore more
            print(f"trial count: {trial_count}")
            if trial_count == 10:
                assert False
        assert False

    def _compute_sliced_volume_inner(
        self, fslset: SuperlevelSet, param: np.ndarray, config: ActiveSamplerConfig
    ) -> float:
        assert config.integration_method == "mc"
        count = 0
        for _ in range(config.n_mc_integral):
            situation = self.situation_sampler()
            x = np.hstack([param, situation])
            if self.fslset.func(np.array([x]))[0] > 0:
                count += 1
        return count / config.n_mc_integral

    def ask(self) -> np.ndarray:
        # COPIED FROM ActiveSamplerBase
        assert self.count_additional == 0
        param_cands, volumes = self._determine_param_candidates()
        assert len(param_cands) > 0
        # update param_best_so_far
        best_param = None
        idx_max_volume = np.argmax(volumes)
        if self.compute_sliced_volume(self.best_param_so_far) < volumes[idx_max_volume]:
            best_param = param_cands[idx_max_volume]
        if best_param is not None:
            self.best_param_so_far += self.config.learning_rate * (
                best_param - self.best_param_so_far
            )
        # FINISH COPYING

        param_metric = self.metric.metirics[0]
        param_center = self.best_param_so_far

        params = []
        while len(params) < 10:
            rand_param = param_metric.generate_random_inball(
                param_center, self.config.n_mc_param_search, self.config.r_exploration
            )[0]
            if self.is_valid_param(rand_param):
                params.append(rand_param)

        x_uncertainty_pairs = []
        for param in params:
            for _ in range(10):
                situation = self.situation_sampler()
                x = np.hstack([param, situation])
                uncertainty = np.min(self.metric(x, self.X))
                x_uncertainty_pairs.append((x, uncertainty))
        x_uncertainty_pairs.sort(key=lambda x: -x[1])
        return x_uncertainty_pairs[0][0]
