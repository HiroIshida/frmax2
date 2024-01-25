import argparse
import json
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import threadpoolctl
import tqdm

from frmax2.bayes_opt import Bound, SaasBoOptimzer
from frmax2.core import (
    BlackBoxSampler,
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
)
from frmax2.environment import AnisoEnvironment

np.random.seed(0)


class Result:
    param_hist: List[np.ndarray]
    size_hist: List[float]
    size_est_hist: List[float]
    n_eval_hist: List[int]
    size_opt_gt: float

    def __init__(self, size_opt_gt: float):
        self.size_opt_gt = size_opt_gt
        self.param_hist = []
        self.size_hist = []
        self.size_est_hist = []
        self.n_eval_hist = []

    def to_json(self) -> str:
        return json.dumps(
            {
                "param_hist": [list(p) for p in self.param_hist],
                "size_hist": self.size_hist,
                "size_est_hist": self.size_est_hist,
                "n_eval_hist": self.n_eval_hist,
                "size_opt_gt": self.size_opt_gt,
            },
            indent=4,
        )

    def dump(self, file_name: str) -> None:
        json_str = self.to_json()
        with open(file_name, "w") as f:
            f.write(json_str)

    @classmethod
    def from_json(cls, json_str: str) -> "Result":
        d = json.loads(json_str)
        result = cls(d["size_opt_gt"])
        result.param_hist = [np.array(p) for p in d["param_hist"]]
        result.size_hist = d["size_hist"]
        result.size_est_hist = d["size_est_hist"]
        result.n_eval_hist = d["n_eval_hist"]
        return result

    @classmethod
    def load(cls, file_name: str) -> "Result":
        with open(file_name, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, help="parameter dim", default=16)
    parser.add_argument("-m", type=str, help="sliced space dim", default=3)
    parser.add_argument("-l", type=int, help="simulation length")
    parser.add_argument("-method", type=str, help="method", default="proposed")
    parser.add_argument("-r", type=float, help="method", default=1.0)
    args = parser.parse_args()
    assert args.method in ["proposed", "bo"]

    env = AnisoEnvironment(int(args.n), int(args.m), with_hollow=True, error_consider_axes=[0, 1])

    param_init = np.zeros(int(args.n))
    param_init[0] = -1.5
    if len(param_init) > 1:
        param_init[1] = -1.5
    if len(param_init) > 2:
        param_init[2] = -1.5
    size_opt = env.evaluate_size(np.zeros(int(args.n)))
    result = Result(size_opt)

    if args.method == "proposed":
        l_iter = 1000 if args.l is None else args.l
        ls_param = np.ones(int(args.n)) * 0.5
        # ls_param = np.ones(int(args.n)) * 1.0
        ls_error = np.ones(int(args.m)) * 0.2
        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        r = round(float(args.r), 1)
        config = DGSamplerConfig(
            param_ls_reduction_rate=1.0,
            n_mc_param_search=30,
            c_svm=10000,
            integration_method="mc",
            n_mc_integral=200,
            r_exploration=r,
            learning_rate=1.0,
        )

        X = []
        Y = []
        n_positive = 0
        n_negative = 0
        n_count_each = 5 * int(args.m)
        while True:
            x = np.hstack([param_init, env.sample_situation()])
            y = env.isInside(x)
            if y:
                n_positive += 1
                if n_positive < n_count_each:
                    X.append(x)
                    Y.append(y)
            else:
                n_negative += 1
                if n_negative < n_count_each:
                    X.append(x)
                    Y.append(y)
            if n_positive >= n_count_each and n_negative >= n_count_each:
                break

        X = np.array(X)
        Y = np.array(Y)
        sampler: BlackBoxSampler = DistributionGuidedSampler(
            X, Y, metric, param_init, config, situation_sampler=env.sample_situation
        )
        size_list = []
        for i in tqdm.tqdm(range(l_iter)):

            with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    x = sampler.ask()
                    y = env.isInside(x)
                    sampler.tell(x, y)

            if i % 20 == 0:
                param_opt = sampler.optimize(200, 0.5, method="cmaes")
                size = env.evaluate_size(param_opt)
                size_est = sampler.compute_sliced_volume(param_opt) * env.sampling_space_volume
                # print(f"param_opt: {param_opt}")
                print(f"rate: {size / size_opt}, rate_est: {size_est / size_opt}")
                size_list.append(size / size_opt)

                result.param_hist.append(param_opt)
                result.size_hist.append(size)
                result.size_est_hist.append(size_est)

                n_eval = len(sampler.X)
                result.n_eval_hist.append(n_eval)

        param_opt = sampler.optimize(200, 0.5, method="cmaes")
        size = env.evaluate_size(param_opt)
        size_est = sampler.compute_sliced_volume(param_opt) * env.sampling_space_volume
        print(f"before additional")
        print(f"param_opt: {param_opt}")
        print(f"rate: {size / size_opt}, rate_est: {size_est / size_opt}")

        # n_additional = args.m * 20
        # for _ in range(n_additional):
        #     x = sampler.ask_additional(param_opt)
        #     y = env.isInside(x)
        #     sampler.tell(x, y)
        #     size = env.evaluate_size(param_opt)
        #     size_est = sampler.compute_sliced_volume(param_opt) * env.sampling_space_volume
        #     result.param_hist.append(param_opt)
        #     result.size_hist.append(size)
        #     result.size_est_hist.append(size_est)
        #     result.n_eval_hist.append(len(sampler.X))

        # size = env.evaluate_size(param_opt)
        # size_est = sampler.compute_sliced_volume(param_opt) * env.sampling_space_volume
        # print(f"after additional")
        # print(f"param_opt: {param_opt}")
        # print(f"rate: {size / size_opt}, rate_est: {size_est / size_opt}")

    else:
        n_eval_count = 0
        prev_samples = None

        def evaluate_volume_mc(param: np.ndarray, n_mc: int = 50) -> float:
            x_list = []
            y_list = []
            for _ in range(n_mc):
                global n_eval_count
                n_eval_count += 1
                s = env.sample_situation()
                x = np.hstack([param, s])
                y = env.isInside(x)
                x_list.append(x)
                y_list.append(y)
            (x_list, y_list)
            y_mean = np.mean(y_list) * env.sampling_space_volume
            return y_mean

        l_iter = 100 if args.l is None else args.l

        X = [param_init]
        Y = [evaluate_volume_mc(param_init)]
        param_bound = Bound(np.ones(int(args.n)) * -2.0, np.ones(int(args.n)) * 2.0)
        bo = SaasBoOptimzer(X, Y, param_bound)

        for _ in range(l_iter):
            x = bo.ask()
            y_est = evaluate_volume_mc(x)
            bo.tell(x, y_est)
            y_real = env.evaluate_size(x)
            print(f"rate: {y_real / size_opt}, rate_est: {y_est / size_opt}")

            result.param_hist.append(x)
            result.size_hist.append(y_real)
            result.size_est_hist.append(y_est)
            result.n_eval_hist.append(n_eval_count)

    # if result directory does not exist create it
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    uuid_str = str(uuid4())  # to identify the result after mc run
    if args.method == "proposed":
        file_name = result_dir / f"proposed_n{args.n}_m{args.m}_r{r}_{uuid_str}.pkl"
    else:
        file_name = result_dir / f"bo_n{args.n}_m{args.m}_{uuid_str}.pkl"
    result.dump(file_name)
