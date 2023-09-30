from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from frmax2.metric import CompositeMetric, Metric
from frmax2.region import FactorizableSuperLevelSet


@dataclass
class ActiveSamplerConfig:
    n_mc_param_search: int = 20
    r_exploration: float = 0.5


class ActiveSampler:
    fslset: FactorizableSuperLevelSet
    is_valid_param: Callable[[np.ndarray], bool]
    config: ActiveSamplerConfig
    best_param_so_far: np.ndarray

    def __init__(
        self,
        fslset: FactorizableSuperLevelSet,
        param_init: np.ndarray,
        config: ActiveSamplerConfig = ActiveSamplerConfig(),
        is_valid_param: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.fslset = fslset
        self.config = config
        self.best_param_so_far = param_init
        if is_valid_param is None:
            self.is_valid_param = lambda x: True

    @property
    def X(self) -> np.ndarray:
        return self.fslset.X

    @property
    def Y(self) -> np.ndarray:
        return self.fslset.Y

    @property
    def axes_param(self) -> np.ndarray:
        return self.fslset.axes_slice

    @property
    def metric(self) -> CompositeMetric:
        return self.fslset.metric

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
                print("sample from ball again as no samples satisfies constraint")

        param_metric = self.metric.metirics[0]
        param_center = self.best_param_so_far

        trial_count = 0
        r = self.config.r_exploration
        while True:
            param_sampled = sample_until_valid(r)
            volumes = np.array([self.fslset.volume_sliced(p) for p in param_sampled])
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
        if self.fslset.volume_sliced(self.best_param_so_far) < volumes[idx_max_volume]:
            self.best_param_so_far = param_cands[idx_max_volume]

        # sample points
        x_best = None
        uncertainty_max = -np.inf
        for param in param_cands:
            region = self.fslset.slice(param)
            if region is not None:
                for co_point in region.points:
                    x = np.hstack([param, co_point])
                    uncertainty = np.min(self.metric(x, self.fslset.X))
                    if uncertainty > uncertainty_max:
                        uncertainty_max = uncertainty
                        x_best = x
        assert x_best is not None
        return x_best

    def tell(self, x: np.ndarray, y: bool, update_clf: bool = True) -> None:
        y_float = 1.0 if y else -1.0

        X = np.vstack([self.X, x])
        Y = np.hstack([self.Y, y_float])

        # update classifier
        if update_clf:
            ls_co = np.sqrt(np.diag(self.metric.metirics[1].cmat))
            param_pre = X[-1][self.axes_param]
            width = self.fslset.region_widths(param_pre)
            ls_co_cand = width * 0.25

            r = 1.5
            ls_co_min = ls_co * (1 / r)
            ls_co_max = ls_co * r
            ls_co = np.max(np.vstack((ls_co_cand, ls_co_min)), axis=0)
            ls_co = np.min(np.vstack((ls_co, ls_co_max)), axis=0)
            metric_co = Metric.from_ls(ls_co)
            new_metric = CompositeMetric([self.metric.metirics[0], metric_co])
        else:
            new_metric = self.metric
        self.fslset = FactorizableSuperLevelSet.fit(
            X, Y, new_metric, n_grid=self.fslset.n_grid, margin=self.fslset.margin
        )
