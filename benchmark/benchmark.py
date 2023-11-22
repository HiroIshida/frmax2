import argparse

import matplotlib.pyplot as plt
import numpy as np
import threadpoolctl
import tqdm

from frmax2.core import (
    BlackBoxSampler,
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
)
from frmax2.environment import GaussianEnvironment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, help="parameter dim", default=10)
    parser.add_argument("-m", type=str, help="sliced space dim", default=5)
    args = parser.parse_args()

    env = GaussianEnvironment(int(args.n), int(args.m), with_hollow=False)

    evaluate_count = 0

    def evaluate(x: np.ndarray) -> bool:
        global evaluate_count
        evaluate_count += 1
        return env.isInside(x)

    n_mc_integral = 2000
    ls_param = np.ones(int(args.n)) * 0.5
    ls_error = np.ones(int(args.m)) * 0.2
    metric = CompositeMetric.from_ls_list([ls_param, ls_error])
    config = DGSamplerConfig(
        param_ls_reduction_rate=0.999,
        n_mc_param_search=30,
        c_svm=10000,
        integration_method="mc",
        n_mc_integral=n_mc_integral,
        r_exploration=0.5,
        learning_rate=1.0,
    )

    def sample_situation() -> np.ndarray:
        return np.random.rand(int(args.m)) * 3.0 - 1.5

    param_init = np.zeros(int(args.n))
    param_init[0] = -0.3
    param_init[1] = -0.5

    X = []
    Y = []
    n_positive = 0
    n_negative = 0
    n_count_each = 5
    while True:
        x = np.hstack([param_init, sample_situation()])
        y = evaluate(x)
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
        X, Y, metric, param_init, config, situation_sampler=sample_situation
    )
    size_list = []
    size_opt = env.evaluate_size(np.zeros(int(args.n)))
    for i in tqdm.tqdm(range(2000)):

        with threadpoolctl.threadpool_limits(limits=1, user_api="openmp"):
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                sampler.update_center()
                x = sampler.ask()
                y = evaluate(x)
                sampler.tell(x, y)

        if i % 20 == 0:
            param_opt = sampler.optimize(200, 0.5, method="cmaes")
            size = env.evaluate_size(param_opt)
            print(size / size_opt)
            size_list.append(size / size_opt)

    plt.plot(size_list)
    plt.show()
