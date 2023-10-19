import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from frmax2.core import DGSamplerConfig, DistributionGuidedSampler
from frmax2.environment import GaussianEnvironment
from frmax2.initialize import initialize
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", type=int, default=201)
args = parser.parse_args()

np.random.seed(args.seed)

create_default_logger(Path("./"), "train", logging.DEBUG)

env = GaussianEnvironment(1, 1, with_bias=True)
ls_param, ls_co = env.default_lengthscales()
ls_param *= 2.0
param_init = env.default_init_param()


env = GaussianEnvironment(1, 1, with_bias=False, with_hollow=True)

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
    epsilon_exploration=0.2
)


def situation_sampler() -> np.ndarray:
    e = np.random.rand() * 3.0 - 1.5
    return np.array([e])

e_list = np.linspace(-0.3, 0.3, 6)
X = np.array([np.hstack([param_init, e]) for e in e_list])
Y = np.array([True if env.isInside(x) else False for x in X])
ls_co = np.array([0.2])
metric = CompositeMetric.from_ls_list([ls_param, ls_co])
sampler = DistributionGuidedSampler(
    X, Y, metric, param_init, situation_sampler=situation_sampler, config=config
)

for i in range(args.n):
    print(i)
    if i % 10 == 0:
        fig, ax = plt.subplots()
        env.visualize_region(-2.5, 1.5, (fig, ax))

        X_positive = sampler.X[sampler.Y]
        X_negative = sampler.X[~sampler.Y]
        ax.scatter(X_positive[:, 0], X_positive[:, 1], c="b")
        ax.scatter(X_negative[:, 0], X_negative[:, 1], c="r")

        xlin = np.linspace(-2.5, 1.5, 100)
        ylin = np.linspace(-1.5, 3.5, 100)
        Xgrid, Ygrid = np.meshgrid(xlin, ylin)
        pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
        values = sampler.fslset.func(pts)
        Z = values.reshape(Xgrid.shape)
        ax.contour(Xgrid, Ygrid, Z, levels=[0], cmpa="jet", zorder=1)
        ax.set_aspect("equal")

        param_best_now = sampler.best_param_so_far
        ax.axvline(x=param_best_now[0], color="k", linestyle="--", zorder=0)

        ax.set_xlim(-2.5, 1.5)
        ax.set_ylim(-1.6, 1.6)
        ax.axhline(y=-1.5, color="k", linestyle="--", zorder=0)
        ax.axhline(y=1.5, color="k", linestyle="--", zorder=0)
        plt.show()

    sampler.update_center()
    x = sampler.ask()
    sampler.tell(x, env.isInside(x))

x_opt = sampler.optimize(100)

for i in range(20):
    x = sampler.ask_additional(x_opt)
    sampler.tell(x, env.isInside(x))
