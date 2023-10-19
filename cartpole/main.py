import argparse
import logging
from hashlib import md5
from pathlib import Path

import dill
import matplotlib.cm as cm
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
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("-n", type=int)
    parser.add_argument("-m", type=int, default=1, help="number of error dim")

    args = parser.parse_args()

    logger = create_default_logger(Path("/tmp/"), "train", logging.INFO)
    param_dof = 7

    file_path = Path(f"./sampler-cache-{args.m}.pkl")
    param_hand = np.array([0.1, 1.0, 1.0, 1.0, 0.5, 0.3, 0.3])

    if args.mode == "train":
        if args.n is None:
            args.n = 1500
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
        fig, ax = plt.subplots(1, 1)
        ax.plot(sampler_cache.best_volume_history, "o-", lw=0.3, ms=2)
        ax.set_xlabel("iteration")
        ax.set_ylabel("weightd volume estimate")
        size = fig.get_size_inches()
        size[0] = size[0] * 0.5
        size[1] = size[0] * 1.2
        fig.set_size_inches(*size)
        # adjust bottom and left
        plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
        plt.grid()
        fig.savefig(f"./volume-history-{args.m}.png", dpi=300)
        plt.show()
        # save fig
        env = Environment(param_dof)

        # get hashvalue of file_path contents
        hash_valie = md5(file_path.read_bytes()).hexdigest()
        test_cache_path = Path(f"./test-cache-{hash_valie}-{args.n}.pkl")

        if test_cache_path.exists():
            with test_cache_path.open(mode="rb") as f:
                e_list, bools, bools_hand = dill.load(f)
        else:
            # benchmark
            # create e points on regular grids
            b_min = -0.75
            b_max = +0.75
            if args.m == 1:
                if args.n is None:
                    args.n = 50
                e_list = np.linspace(b_min, b_max, args.n).reshape(-1, 1)
            elif args.m == 2:
                if args.n is None:
                    args.n = 20
                axis_list = [np.linspace(b_min, b_max, args.n)] * 2
                e1_grid, e2_grid = np.meshgrid(*axis_list)
                e_list = np.hstack([e1_grid.reshape(-1, 1), e2_grid.reshape(-1, 1)])
            elif args.m == 3:
                if args.n is None:
                    args.n = 10
                axis_list = [np.linspace(b_min, b_max, args.n)] * 3
                e1_grid, e2_grid, e3_grid = np.meshgrid(*axis_list)
                e_list = np.hstack(
                    [e1_grid.reshape(-1, 1), e2_grid.reshape(-1, 1), e3_grid.reshape(-1, 1)]
                )
            else:
                assert False

            bools = []
            bools_hand = []
            for e in tqdm.tqdm(e_list):
                res = env._rollout(param_opt, e)
                res_hand = env._rollout(param_hand, e)
                bools.append(res)
                bools_hand.append(res_hand)
            bools = np.array(bools)
            bools_hand = np.array(bools_hand)
            with test_cache_path.open(mode="wb") as f:
                dill.dump((e_list, bools, bools_hand), f)
        print(f"hand volume: {np.sum(bools_hand) / len(bools_hand)}")
        print(f"actual volume: {np.sum(bools) / len(bools)}")

        if args.m == 1:
            es = np.linspace(min(e_list), max(e_list), 3000)
            decision_values = [
                sampler.fslset.func(np.array([np.hstack([param_opt, e])])) for e in es
            ]
            cross_points = es[np.abs(decision_values) < 0.1]

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(e_list[bools_hand], 10 * np.ones_like(e_list[bools_hand]), "o", c="blue", ms=2)
            ax1.plot(
                e_list[~bools_hand], -10 * np.ones_like(e_list[~bools_hand]), "o", c="red", ms=2
            )
            ax1.axhline(y=0.0, color="k", linestyle="-")

            ax2.plot(e_list[bools], 10 * np.ones_like(e_list[bools]), "o", c="blue", ms=2)
            ax2.plot(e_list[~bools], -10 * np.ones_like(e_list[~bools]), "o", c="red", ms=2)
            ax2.axhline(y=0.0, color="k", linestyle="-")
            ax2.plot(es, decision_values, "-", lw=2.0, c="gray")
            # show cross points by vertical lines
            for x in cross_points:
                ax2.axvline(x=x, color="k", linestyle="--")

            # save fig
            size = fig.get_size_inches()
            size[0] = size[0] * 1.0
            size[1] = size[0] * 0.8
            fig.set_size_inches(*size)
            plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
            plt.grid()
            fig.savefig(f"./comparison-{args.m}.png", dpi=300)
            plt.show()
        elif args.m == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            E = np.array(e_list)
            bools = np.array(bools)
            bools_hand = np.array(bools_hand)

            s = 5
            ax1.scatter(E[bools_hand, 1], E[bools_hand, 0], c="blue", s=s)
            ax1.scatter(E[~bools_hand, 1], E[~bools_hand, 0], c="red", s=s)

            cmap = cm.get_cmap("coolwarm").reversed()
            sampler.fslset.show_sliced(
                param_opt,
                list(range(len(param_opt))),
                50,
                (fig, ax2),
                rich=True,
                cmap=cmap,
                levels=20,
            )
            ax2.scatter(E[bools, 1], E[bools, 0], c="blue", s=s)
            ax2.scatter(E[~bools, 1], E[~bools, 0], c="red", s=s)

            for ax in (ax1, ax2):
                ax.set_aspect("equal")
                ax.set_xlim(-0.8, 0.8)
                ax.set_ylim(-0.8, 0.8)
            # remove y memori of ax2
            ax2.set_yticks([])
            # save fig
            size = fig.get_size_inches()
            size[0] = size[0] * 1.0
            size[1] = size[0] * 0.5
            fig.set_size_inches(*size)
            plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
            fig.savefig(f"./comparison-{args.m}.png", dpi=300)
            plt.show()
        elif args.m == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            print("do nothing")
        else:
            assert False
