import argparse
import logging
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from common import Environment

from frmax2.core import DGSamplerConfig, DistributionGuidedSampler, SamplerCache
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger

if __name__ == "__main__":
    # argparse to select mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("-m", type=int, default=1, help="number of error dim")

    args = parser.parse_args()

    logger = create_default_logger(Path("/tmp/"), "train", logging.INFO)
    param_dof = 7

    file_path = Path(f"./sampler-cache-{args.m}.pkl")
    param_hand = np.array([0.1, 1.0, 1.0, 1.0, 0.5, 0.3, 0.3])

    if args.mode == "train":
        if args.m == 1:
            n_mc_integral = 80
            learning_rate = 0.2
            r_exploration = 2.0
            ls_error = np.array([0.02] * args.m)
            n_init_sample = 20
            n_add_sample = 20
        elif args.m == 2:
            n_mc_integral = 120
            learning_rate = 0.05
            r_exploration = 2.0
            ls_error = np.array([0.03] * args.m)
            n_init_sample = 40
            n_add_sample = 40
        else:
            n_mc_integral = 300
            learning_rate = 0.05
            # r_exploration = 1.2
            r_exploration = 2.0
            ls_error = np.array([0.05] * args.m)
            n_init_sample = 60
            n_add_sample = 60

        np.random.seed(1)
        env = Environment(param_dof)
        param_init = param_hand
        ls_param = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1])
        print(f"param_init: {param_init.shape}")

        X = []
        Y = []
        for i in tqdm.tqdm(range(n_init_sample)):
            e = -np.ones(args.m) * 0.5 + np.random.rand(args.m)
            res = env._rollout(param_init, e)
            Y.append(res)
            X.append(np.hstack([param_init, e]))
        X = np.array(X)
        Y = np.array(Y)
        assert sum(Y) > 0
        logger.info(f"initial volume: {np.sum(Y) / len(Y)}")

        def situation_sampler() -> np.ndarray:
            w = 1.5
            e = np.random.rand(args.m) * w - 0.5 * w
            return np.array(e)

        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        config = DGSamplerConfig(
            n_mc_param_search=10,
            c_svm=1000000,
            integration_method="mc",
            n_mc_integral=n_mc_integral,
            r_exploration=r_exploration,
            learning_rate=learning_rate,
            param_ls_reduction_rate=0.998,
        )

        def is_valid_param(param: np.ndarray) -> bool:
            (
                energy_alpha,
                lqr_q1,
                lqr_q2,
                lqr_q3,
                lqr_r,
                switch_coditing_theta,
                switch_coditing_theta_dot,
            ) = param
            if energy_alpha < 0.0:
                return False
            eps = 0.05
            if lqr_q1 < eps or lqr_q2 < eps or lqr_q3 < eps or lqr_r < eps:
                return False
            if switch_coditing_theta < 0.0 or switch_coditing_theta_dot < 0.0:
                return False
            return True

        sampler = DistributionGuidedSampler(
            X,
            Y,
            metric,
            param_init,
            situation_sampler=situation_sampler,
            config=config,
            is_valid_param=is_valid_param,
        )

        for i in tqdm.tqdm(range(args.n)):
            sampler.update_center()
            x = sampler.ask()
            sampler.tell(x, env.rollout(x))
            if i % 50 == 0:
                # save sampler
                param_opt = sampler.optimize(300)
                with file_path.open(mode="wb") as f:
                    dill.dump((sampler, param_opt), f)

        param_opt = sampler.optimize(300)

        for i in range(n_add_sample):
            x = sampler.ask_additional(param_opt)
            sampler.tell(x, env.rollout(x))
        with file_path.open(mode="wb") as f:
            dill.dump((sampler, param_opt), f)

    elif args.mode == "test":
        # load sampler
        with file_path.open(mode="rb") as f:
            sampler, param_opt = dill.load(f)
            sampler_cache: SamplerCache = sampler.sampler_cache
        plt.plot(sampler_cache.best_volume_history)
        plt.show()

        env = Environment(param_dof)

        # benchmark
        e_list = []
        bools = []
        bools_hand = []
        for i in tqdm.tqdm(range(300)):
            e = sampler.situation_sampler()
            res = env._rollout(param_opt, e)
            res_hand = env._rollout(param_hand, e)
            e_list.append(e)
            bools.append(res)
            bools_hand.append(res_hand)
        print(f"hand volume: {np.sum(bools_hand) / len(bools_hand)}")
        print(f"actual volume: {np.sum(bools) / len(bools)}")

        if args.m == 1:
            es = np.linspace(min(e_list), max(e_list), 100)
            decision_values = [
                sampler.fslset.func(np.array([np.hstack([param_opt, e])])) for e in es
            ]
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(e_list, bools, "o")
            ax1.plot(es, decision_values, "-")
            ax2.plot(e_list, bools_hand, "o")
            plt.show()
        elif args.m == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            E = np.array(e_list)
            bools = np.array(bools)
            bools_hand = np.array(bools_hand)
            ax1.scatter(E[bools, 1], E[bools, 0], c="blue")
            ax1.scatter(E[~bools, 1], E[~bools, 0], c="red")
            sampler.fslset.show_sliced(param_opt, list(range(len(param_opt))), 50, (fig, ax1))
            ax2.scatter(E[bools_hand, 1], E[bools_hand, 0], c="blue")
            ax2.scatter(E[~bools_hand, 1], E[~bools_hand, 0], c="red")
            plt.show()
        elif args.m == 3:
            print("do nothing")
        else:
            assert False
