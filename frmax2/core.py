import copy
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Optional, Tuple, TypeVar, Union

import dill
import numpy as np
import threadpoolctl
from cmaes import CMA

from frmax2.metric import CompositeMetric, Metric
from frmax2.region import SuperlevelSet, get_co_axes

logger = logging.getLogger(__name__)

ActiveSamplerConfigT = TypeVar("ActiveSamplerConfigT", bound="ActiveSamplerConfig")
SituationSamplerT = TypeVar("SituationSamplerT", bound=Union[Callable[[], np.ndarray], None])


class BlackBoxSampler(ABC):
    @abstractmethod
    def ask(self) -> np.ndarray:
        pass

    @abstractmethod
    def tell(self, x: np.ndarray, y: bool) -> None:
        pass


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
    param_ls_reduction_rate: float = 1.0  # only used in DistributionGuidedSampler
    integration_method: Literal["mc", "grid"] = "grid"
    measure_width_method: Literal["mc", "grid"] = "grid"
    sample_error_method: Literal["mc-margin", "mc-inside", "grid"] = "grid"
    box_cut: bool = False


@dataclass
class DGSamplerConfig(ActiveSamplerConfig):
    n_mc_uncertainty_search: int = 100
    epsilon_exploration: float = 0.1


class SamplerCache:
    best_param_history: List[np.ndarray]
    best_volume_history: List[float]

    def __init__(self):
        self.best_param_history = []
        self.best_volume_history = []


class PreFactoBranchedAskResultCaliculator:
    """
    A class for optimizing the active sampling procedure in contexts where the evaluation of
    sample outcomes (y values) is time-intensive.

    This class is particularly beneficial in scenarios like robotic reinforcement learning or
    experimental design, where computing the y value corresponding to a sample (x) involves
    time-consuming processes such as executing a task with a real robot or conducting a
    physical experiment. It leverages multiprocessing to parallelize the computation of the
    next sampling point (ask() method) while the y value is being evaluated, thus reducing
    overall waiting time in the sampling loop.
    """

    def __init__(self):
        self.mp_queue = None
        self.mp_process = None

    def start_bifurcation_prediction(self, sampler: "ActiveSamplerBase", x_asked: np.ndarray):
        # run another process to predict background
        self.mp_queue = mp.Queue()
        self.mp_process = mp.Process(
            target=self._worker, args=(dill.dumps(sampler), x_asked, self.mp_queue)
        )
        self.mp_process.start()

    def get_bifurcation_prediction(
        self, y_observed: bool
    ) -> Optional[Tuple["ActiveSamplerBase", np.ndarray]]:
        if self.mp_process is None:
            return None
        assert self.mp_queue is not None

        self.mp_process.join()
        ret_if_true, ret_if_false = self.mp_queue.get()
        self.mp_queue = None
        self.mp_process = None
        if y_observed:
            ret = ret_if_true
        else:
            ret = ret_if_false
        sampler_dilled, x_predicted = ret
        sampler = dill.loads(sampler_dilled)
        return sampler, x_predicted

    @staticmethod
    def _worker(sampler_dilled: bytes, x_asked: np.ndarray, mp_queue) -> None:
        sampler_hypo_if_true = dill.loads(sampler_dilled)
        sampler_hypo_if_true.tell(x_asked, True)
        x_predicted_if_true = sampler_hypo_if_true._ask()

        sampler_hypo_if_false = dill.loads(sampler_dilled)
        sampler_hypo_if_false.tell(x_asked, False)
        x_predicted_if_false = sampler_hypo_if_false._ask()

        mp_queue.put(
            (
                (dill.dumps(sampler_hypo_if_true), x_predicted_if_true),
                (dill.dumps(sampler_hypo_if_false), x_predicted_if_false),
            )
        )


class NullIsVaildParam:
    def __call__(self, x: np.ndarray) -> bool:
        return True


@dataclass
class UniformSituationSampler:
    lb: np.ndarray
    ub: np.ndarray

    def __call__(self) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub)


class ActiveSamplerBase(BlackBoxSampler, Generic[ActiveSamplerConfigT, SituationSamplerT]):
    fslset: SuperlevelSet
    metric: CompositeMetric
    is_valid_param: Callable[[np.ndarray], bool]
    config: ActiveSamplerConfigT
    best_param_so_far: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    axes_param: List[int]
    sampler_cache: SamplerCache
    count_additional: int
    c_svm_current: float
    situation_sampler: SituationSamplerT
    prefact_ask_calc: Optional[PreFactoBranchedAskResultCaliculator]

    def __init__(
        self,
        X: np.ndarray,  # float
        Y: np.ndarray,  # bool
        metric: CompositeMetric,
        param_init: np.ndarray,
        config: ActiveSamplerConfigT,
        is_valid_param: Optional[Callable[[np.ndarray], bool]] = None,
        situation_sampler: SituationSamplerT = None,
        use_prefacto_branched_ask: bool = False,
    ):
        c_svm_current = config.c_svm
        slset = SuperlevelSet.fit(X, Y, metric, C=c_svm_current, box_cut=config.box_cut)
        self.fslset = slset
        self.metric = metric
        self.config = config
        self.best_param_so_far = param_init
        if is_valid_param is None:
            is_valid_param = NullIsVaildParam()
        self.is_valid_param = is_valid_param
        self.X = X
        self.Y = Y
        self.axes_param = list(range(len(param_init)))
        self.sampler_cache = SamplerCache()
        self.c_svm_current = c_svm_current
        self.count_additional = 0
        self.situation_sampler = situation_sampler
        if use_prefacto_branched_ask:
            self.prefact_ask_calc = PreFactoBranchedAskResultCaliculator()
        else:
            self.prefact_ask_calc = None

    def update_center(self):
        raise NotImplementedError("this function is delted")

    @staticmethod
    @abstractmethod
    def _compute_sliced_volume_inner(
        fslset: SuperlevelSet, param: np.ndarray, config: ActiveSamplerConfigT
    ) -> float:
        pass

    def optimize(
        self,
        n_search: int,
        r_search: float = 0.5,
        method: Literal["random", "cmaes"] = "random",
    ) -> np.ndarray:
        center = self.best_param_so_far
        param_metric = self.metric.metirics[0]

        if method == "random":
            rand_params = param_metric.generate_random_inball(center, n_search, r_search)
            rand_params_filtered = list(filter(self.is_valid_param, rand_params))
            volumes = self.compute_sliced_volume_batch(rand_params_filtered)
            idx_max_volume = np.argmax(volumes)
            best_param_so_far = rand_params_filtered[idx_max_volume]
            return best_param_so_far
        elif method == "cmaes":
            ls_co = np.sqrt(np.diag(param_metric.cmat))
            assert np.all(ls_co == ls_co[0]), "fix this"
            optimizer = CMA(mean=center, sigma=ls_co[0] * r_search)
            n_count_evaluate = 0
            best_param = center
            best_volume = self.compute_sliced_volume_batch([best_param])[0]
            while n_count_evaluate < n_search:
                dataset = []
                for _ in range(optimizer.population_size):
                    param = optimizer.ask()
                    val = self.compute_sliced_volume_batch([param])[0]
                    dataset.append((param, -val))  # minus for minimization
                    n_count_evaluate += 1
                optimizer.tell(dataset)
                if optimizer.should_stop() or n_count_evaluate >= n_search:
                    break
                params, negative_volumes = zip(*dataset)
                best_index = np.argmin(negative_volumes)

                if -negative_volumes[best_index] > best_volume:
                    best_param = params[best_index]
                    best_volume = -negative_volumes[best_index]

            return best_param
        else:
            assert False

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

    def ask(self) -> np.ndarray:
        if self.prefact_ask_calc is None:
            return self._ask()
        else:
            ret = self.prefact_ask_calc.get_bifurcation_prediction(bool(self.Y[-1]))
            bifurcation_prediction_not_called = ret is None
            if bifurcation_prediction_not_called:
                x = self._ask()
            else:
                self = ret[0]  # so dirty
                x = ret[1]
            self.prefact_ask_calc.start_bifurcation_prediction(self, x)
            return x

    @abstractmethod
    def _ask(self) -> np.ndarray:
        pass

    def __getstate__(self):  # pickling
        state = self.__dict__.copy()
        state["prefact_ask_calc"] = self.prefact_ask_calc is not None
        return state

    def __setstate__(self, state):  # unpickling
        has_prefact_ask_calc = state["prefact_ask_calc"]
        if has_prefact_ask_calc:
            state["prefact_ask_calc"] = PreFactoBranchedAskResultCaliculator()
        else:
            state["prefact_ask_calc"] = None
        self.__dict__.update(state)


class HolllessActiveSampler(ActiveSamplerBase[ActiveSamplerConfig, None]):
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

        self.sampler_cache.best_param_history.append(copy.deepcopy(param_center))
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
        x_uncertainty_pairs.sort(key=lambda x: -x[1])
        return x_uncertainty_pairs[0][0]

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


class DistributionGuidedSampler(ActiveSamplerBase[DGSamplerConfig, Callable[[], np.ndarray]]):
    X_cand_sorted_cache: np.ndarray  # for later visualization

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
        assert X.ndim == 2
        self.c_svm_current = max(
            self.config.c_svm_min, self.c_svm_current * self.config.c_svm_reduction_rate
        )
        logger.debug(f"current c_svm: {self.c_svm_current}")

        # only here is different from ActiveSamplerBase
        metric_param = self.metric.metirics[0]
        mat = metric_param.cmat
        ls_param = np.sqrt(np.diag(mat))
        ls_param = ls_param * self.config.param_ls_reduction_rate
        logger.info(f"current ls_param: {ls_param}")
        metric_param = Metric.from_ls(ls_param)
        new_metric = CompositeMetric([metric_param, self.metric.metirics[1]])
        self.metric = new_metric

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

    def _determine_param_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        # perfectly copied from ActiveSamplerBase

        def sample_until_valid(r_param: float) -> np.ndarray:
            param_metric = self.metric.metirics[0]
            param_center = self.best_param_so_far
            while True:
                rand_params = param_metric.generate_random_inball(
                    param_center, self.config.n_mc_param_search, r_param
                )
                filtered = list(filter(self.is_valid_param, rand_params))
                if len(filtered) > 0:
                    return np.array(filtered)
                logger.debug("sample from ball again as no samples satisfies constraint")

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
        x_list = []
        for _ in range(config.n_mc_integral):
            situation = self.situation_sampler()
            x = np.hstack([param, situation])
            x_list.append(x)
        X = np.array(x_list)

        with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
            count = np.sum(self.fslset.func(X) > 0)
        return count / config.n_mc_integral

    def _update_center(self):
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
        best_volume_guess = self.compute_sliced_volume(self.best_param_so_far)
        logger.info(f"current best param: {self.best_param_so_far}")
        logger.info(f"current best volume: {best_volume_guess}")
        self.sampler_cache.best_param_history.append(copy.deepcopy(self.best_param_so_far))
        self.sampler_cache.best_volume_history.append(best_volume_guess)

    def _ask(self, mode: Optional[Literal["exploration", "exploitation"]] = None) -> np.ndarray:

        self._update_center()

        param_metric = self.metric.metirics[0]
        param_center = self.best_param_so_far
        if mode is None:
            do_exploitition = np.random.rand() > self.config.epsilon_exploration
        elif mode == "exploration":
            do_exploitition = False
        elif mode == "exploitation":
            do_exploitition = True
        else:
            assert False

        N_batch = 200
        X_cand = []
        while len(X_cand) < self.config.n_mc_uncertainty_search:
            rand_params = param_metric.generate_random_inball(
                param_center, N_batch, self.config.r_exploration
            )
            rand_params_filtered = list(filter(self.is_valid_param, rand_params))
            X_cand_cand = [np.hstack([x, self.situation_sampler()]) for x in rand_params_filtered]
            values = self.fslset.func(np.array(X_cand_cand))
            inside_svm_margin = np.logical_and(values > 0, values < 1.0)
            for x, inside in zip(X_cand_cand, inside_svm_margin):
                if (not do_exploitition) or inside:
                    X_cand.append(x)
        X_cand = np.array(X_cand)[: self.config.n_mc_uncertainty_search]

        def uncertainty(x: np.ndarray) -> float:
            return np.min(self.metric(x, self.X))

        X_cand_sorted: np.ndarray = sorted(X_cand, key=lambda x: -uncertainty(x))
        self.X_cand_sorted_cache = np.array(X_cand_sorted)
        x_most_uncertain = X_cand_sorted[0]
        return x_most_uncertain

    def ask_additional(self, param_here: np.ndarray) -> np.ndarray:
        sliced_points = []
        while len(sliced_points) < 100:
            e = self.situation_sampler()
            pt = np.hstack([param_here, e])
            value = self.fslset.func(np.array([pt]))
            inside_svm_margin = value > 0 and value < 1.0
            if inside_svm_margin:
                sliced_points.append(e)
        sliced_points = np.array(sliced_points)

        # START COPYING FROM ActiveSamplerBase >>
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
        # << FINISH COPYING FROM ActiveSamplerBase
        return np.hstack([param_here, e_new])

    def confidence_score(self, param, N_mc: int = 3000) -> float:
        e_list = [self.situation_sampler() for _ in range(N_mc)]
        X = np.array([np.hstack([param, e]) for e in e_list])
        values = self.fslset.func(X)
        inside_svm_margin = np.logical_and(values > 0, values < 1.0)
        rate_inside = np.mean(inside_svm_margin)
        return 1 - rate_inside

    def get_com(self, param_here: np.ndarray, n_mc: int = 2000) -> np.ndarray:
        assert param_here.ndim == 1
        error_dim = self.metric.metirics[1].dim
        X = np.array([np.hstack([param_here, self.situation_sampler()]) for _ in range(n_mc)])
        bools = self.fslset.func(X) > 0
        E_positive = X[bools, -error_dim:]
        e_mean = np.mean(E_positive, axis=0)
        return e_mean

    def get_optimal_after_additional(self) -> np.ndarray:
        assert self.count_additional > 0
        return self.X[-1][np.array(self.axes_param)]
