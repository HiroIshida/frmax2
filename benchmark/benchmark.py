import argparse
import json
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import threadpoolctl
import tqdm
from cmaes import CMA

from frmax2.bayes_opt import Bound, SaasBoOptimzer
from frmax2.core import (
    BlackBoxSampler,
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
)
from frmax2.environment import AnisoEnvironment


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
    parser.add_argument("-n", type=int, help="parameter dim", default=16)
    parser.add_argument("-m", type=int, help="sliced space dim", default=3)
    parser.add_argument("-l", type=int, help="simulation length")
    parser.add_argument("-method", type=str, help="method", default="proposed")
    parser.add_argument("-c", type=float, help="method", default=10000)
    args = parser.parse_args()
    assert args.method in ["proposed", "bo", "cmaes"]

    env = AnisoEnvironment(
        int(args.n), int(args.m), with_hollow=True, error_consider_axes=[0, 1], random_basis=True
    )

    param_init = np.zeros(int(args.n))
    param_init[0] = -1.0
    param_init = np.linalg.inv(env.M).dot(param_init)  # because of random basis
    size_opt = env.evaluate_size(np.zeros(int(args.n)))
    result = Result(size_opt)

    print(f"size now {env.evaluate_size(param_init)}")

    if args.method == "proposed":
        l_iter = 1000 if args.l is None else args.l
        ls_param = np.ones(int(args.n)) * 0.5
        ls_error = np.ones(int(args.m)) * 0.2
        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        config = DGSamplerConfig(
            param_ls_reduction_rate=1.0,
            n_mc_param_search=30,
            c_svm=args.c,
            integration_method="mc",
            n_mc_integral=200,
            r_exploration=1.0,
            learning_rate=0.5,
        )
        param_metric = metric.metirics[0]

        X = []
        Y = []
        n_init_sample = int(args.m) * 10
        for _ in range(n_init_sample):
            e = env.sample_situation()
            x = np.hstack([param_init, e])
            y = env.isInside(x)
            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        sampler: BlackBoxSampler = DistributionGuidedSampler(
            X, Y, metric, param_init, config, situation_sampler=env.sample_situation
        )

        result.param_hist.append(param_init)
        result.size_hist.append(env.evaluate_size(param_init))
        size_est = sampler.compute_sliced_volume(param_init) * env.sampling_space_volume
        result.size_est_hist.append(size_est)
        result.n_eval_hist.append(n_init_sample)

        for i in tqdm.tqdm(range(l_iter)):

            with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
                with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                    x = sampler.ask()
                    y = env.isInside(x)
                    sampler.tell(x, y)

            if i % 20 == 19:
                param_opt = sampler.optimize(200, 0.5, method="cmaes")
                size = env.evaluate_size(param_opt)
                size_est = sampler.compute_sliced_volume(param_opt) * env.sampling_space_volume
                print(f"rate: {size / size_opt}, rate_est: {size_est / size_opt}")

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
    else:
        n_eval_count = 0

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

        if args.method == "bo":
            X = [param_init]
            Y = [evaluate_volume_mc(param_init)]
            param_bound = Bound(np.ones(int(args.n)) * -2.0, np.ones(int(args.n)) * 2.0)
            bo = SaasBoOptimzer(X, Y, param_bound)

            result.param_hist.append(param_init)
            result.size_hist.append(env.evaluate_size(param_init))
            result.size_est_hist.append(Y[0])
            result.n_eval_hist.append(n_eval_count)

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
        elif args.method == "cmaes":
            alpha = 1.0
            population_size = int((4 + np.floor(np.log(args.n) * 3).astype(int)) * alpha)
            bounds = np.array([(-2.0, 2.0) for _ in range(int(args.n))])
            optimizer = CMA(
                mean=param_init, sigma=0.5, population_size=population_size, bounds=bounds
            )

            n_count_evaluate = 0
            best_param = param_init
            best_volume = evaluate_volume_mc(param_init)
            print(f"population size: {optimizer.population_size}")
            while True:
                dataset = []
                for _ in range(optimizer.population_size):
                    param = optimizer.ask()
                    val = evaluate_volume_mc(param)
                    dataset.append((param, -val))  # minus for minimization
                    n_count_evaluate += 1
                optimizer.tell(dataset)
                if optimizer.should_stop() or n_count_evaluate >= l_iter:
                    break
                params, negative_volumes = zip(*dataset)
                best_index = np.argmin(negative_volumes)
                param, negvol = dataset[best_index]
                est_volume = -negvol
                real_volume = env.evaluate_size(param)
                result.param_hist.append(param)
                result.size_hist.append(real_volume)
                result.size_est_hist.append(est_volume)
                result.n_eval_hist.append(n_count_evaluate)
                print(f"rate: {real_volume / size_opt}, rate_est: {est_volume / size_opt}")
        else:
            assert False

    # if result directory does not exist create it
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    uuid_str = str(uuid4())  # to identify the result after mc run
    file_name = f"{args.method}_n{args.n}_m{args.m}_{uuid_str}.json"

    file_path = result_dir / file_name
    result.dump(file_path)
