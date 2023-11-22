import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frmax2.core import (
    ActiveSamplerConfig,
    DGSamplerConfig,
    DistributionGuidedSampler,
    HolllessActiveSampler,
)
from frmax2.environment import GaussianEnvironment
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger, temp_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", type=int, default=201)
parser.add_argument("-m", type=int, default=1)
parser.add_argument("-eps", type=float, default=0.4)
parser.add_argument("--mode", type=str, default="default")  # default, exploration, exploitition
parser.add_argument("--hollow", action="store_true", help="hollow")
parser.add_argument("--old", action="store_true", help="use old algorithm")

args = parser.parse_args()

np.random.seed(args.seed)

create_default_logger(Path("./"), "train", logging.INFO)

env = GaussianEnvironment(1, args.m, with_bias=False, with_hollow=args.hollow)
ls_param, ls_co = env.default_lengthscales()
param_init = env.default_init_param()
param_init[0] = -1.0

error_dim = args.m


def situation_sampler() -> np.ndarray:
    e = np.random.rand(error_dim) * 3.0 - 1.5
    return e


# e_list = np.linspace(-0.3, 0.3, 6)
e_list = np.array([situation_sampler() for _ in range(100)])
X = np.array([np.hstack([param_init, e]) for e in e_list])
Y = np.array([True if env.isInside(x) else False for x in X])
ls_co = np.array([0.2])
metric = CompositeMetric.from_ls_list([ls_param, ls_co])

if args.old:
    ls_param *= 2.0
    config = ActiveSamplerConfig(n_mc_param_search=100, n_grid=30, box_cut=False, c_svm=100.0)
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)
else:
    ls_param *= 3.0
    config = DGSamplerConfig(
        n_mc_param_search=20,
        n_grid=30,
        box_cut=False,
        c_svm=10.0,
        integration_method="mc",
        n_mc_integral=20,
        c_svm_reduction_rate=1.0,
        r_exploration=0.5,
        learning_rate=0.5,
        epsilon_exploration=args.eps,
    )
    sampler = DistributionGuidedSampler(
        X, Y, metric, param_init, situation_sampler=situation_sampler, config=config
    )

save_dir = "./figs_extended/"

for i in range(args.n):
    print(i)
    if i % 1 == 0:
        fig, ax = plt.subplots()
        env.visualize_region(-2.5, 1.5, (fig, ax))

        X_positive = sampler.X[sampler.Y]
        X_negative = sampler.X[~sampler.Y]
        ax.scatter(X_positive[:, 0], X_positive[:, 1], c="b", s=10)
        ax.scatter(X_negative[:, 0], X_negative[:, 1], c="r", s=10)

        xlin = np.linspace(-2.5, 1.5, 100)
        ylin = np.linspace(-1.5, 3.5, 100)
        Xgrid, Ygrid = np.meshgrid(xlin, ylin)
        pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
        values = sampler.fslset.func(pts)
        Z = values.reshape(Xgrid.shape)
        ax.contour(Xgrid, Ygrid, Z, levels=[0], cmap="jet", zorder=1)
        ax.set_aspect("equal")

        param_best_now = sampler.best_param_so_far
        ax.axvline(x=param_best_now[0], color="k", linestyle="--", zorder=0)
        ax.axvspan(
            sampler.best_param_so_far - ls_param.item() * config.r_exploration,
            sampler.best_param_so_far + ls_param.item() * config.r_exploration,
            ymin=0.0,
            ymax=2.5,
            alpha=0.1,
            color="yellow",
        )

        if args.mode != "default":
            assert isinstance(sampler, DistributionGuidedSampler)
            assert args.mode in ["exploration", "exploitation"]
            with temp_seed(0, True):
                sampler.ask(mode=args.mode)
            ax.scatter(
                sampler.X_cand_sorted_cache[:, 0], sampler.X_cand_sorted_cache[:, 1], c="k", s=1
            )
            x_selected = sampler.X_cand_sorted_cache[0]
            ax.scatter(x_selected[0], x_selected[1], c="orange", marker="*", s=100)
        else:
            assert args.mode == "default"

        ax.set_xlim(-2.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        # ax.axhline(y=-1.5, color="k", linestyle="--", zorder=0)
        # ax.axhline(y=1.5, color="k", linestyle="--", zorder=0)
        fig.set_size_inches(5.0, 3.5)
        base_name = "toy_example" if args.old else "toy_example_extended"
        if args.hollow:
            file_name = f"{base_name}_hollow_{i}.png"
        else:
            file_name = f"{base_name}_{i}.png"

        if args.mode != "default":
            file_name = file_name.replace(".png", f"_{args.mode}.png")
        file_name_full = save_dir + file_name
        plt.savefig(file_name_full, dpi=300)
        plt.close()

        # plot coverage curve
        if args.mode == "default":
            fig, ax = plt.subplots()
            param_lin = np.linspace(-2.5, 1.5, 100)
            volumes = [
                sampler.fslset.sliced_volume_grid_points(np.array([p]), [0], 1000) / 3.0
                for p in param_lin
            ]
            ax.plot(xlin, [env.evaluate_size(np.array([p])) / 3.0 for p in param_lin], "gray")
            ax.plot(param_lin, volumes, "b")
            ax.set_xlabel("param")
            ax.set_ylabel("coverage")
            ax.set_xlim(-2.5, 1.5)
            ax.set_ylim(0.0, 0.8)
            fig.set_size_inches(5.0, 2.0)
            if args.hollow:
                plt.savefig(save_dir + f"toy_example_extended_coverage_hollow_{i}.png", dpi=300)
            else:
                plt.savefig(save_dir + f"toy_example_extended_coverage_{i}.png", dpi=300)
            plt.close()

    if isinstance(sampler, DistributionGuidedSampler):
        sampler.update_center()
    x = sampler.ask()
    sampler.tell(x, env.isInside(x))

# x_opt = sampler.optimize(100)

# for i in range(20):
#     x = sampler.ask_additional(x_opt)
#     sampler.tell(x, env.isInside(x))
