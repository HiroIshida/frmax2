import argparse

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from frmax2.environment import GaussianEnvironment

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", type=int, default=70)
args = parser.parse_args()

np.random.seed(args.seed)

env = GaussianEnvironment(1, 1, with_bias=False)
param_init = env.default_init_param()[0]

X = []
Y = []


def black_box_function(param):
    N_mc = 30
    count = 0
    for e in np.linspace(-1.5, 1.5, N_mc):
        x = np.hstack([param, e])
        y = env.isInside(x)
        if y:
            count += 1
        X.append(x)
        Y.append(y)
    return count / N_mc


pbounds = {"param": (-2.5, 1.4)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)
# optimizer.set_gp_params(n_restarts_optimizer=)
utility = UtilityFunction()

save_dir = "./figs_bayes/"
show_gp_result = False
show_sample = False

for i in range(10):
    if i > 0:
        next_point = optimizer.suggest(utility)
    else:
        next_point = {"param": param_init}
        # next_point = optimizer.suggest(utility)

    print(next_point)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    fig, ax = plt.subplots(1, 1)
    xlin = np.linspace(-2.5, 1.5, 100)
    ax.plot(xlin, [env.evaluate_size(np.array([x])) / 3.0 for x in xlin], "gray")

    if show_gp_result:
        gp = optimizer._gp
        mu, sigma = gp.predict(xlin.reshape(-1, 1), return_std=True)
        ax.plot(xlin, mu, "blue")
        ax.fill_between(
            xlin, mu - 2 * sigma, mu + 2 * sigma, color="b", alpha=0.2, label=r"$2\sigma(x)$"
        )
        ax.plot(optimizer.space.params.flatten(), optimizer.space.target, "k.", markersize=10)

    ax.set_ylim(-2.5, 1.5)
    ax.set_ylim(0, 0.8)
    fig.set_size_inches(5.0, 2.0)
    plt.savefig(save_dir + f"bayes_{i}.png")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    env.visualize_region(-2.5, 1.5, (fig, ax))
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    if show_sample:
        ax.scatter(X_arr[Y_arr, 0], X_arr[Y_arr, 1], c="b", s=2)
        ax.scatter(X_arr[~Y_arr, 0], X_arr[~Y_arr, 1], c="r", s=2)
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.set_size_inches(5.0, 3.5)
    plt.savefig(save_dir + f"bayes_{i}_sample.png")
    # plt.show()
