import argparse

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from frmax2.core import ActiveSamplerConfig, HolllessActiveSampler
from frmax2.environment import GaussianEnvironment
from frmax2.initialize import initialize
from frmax2.metric import CompositeMetric

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="hollless")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cache", action="store_true")

parser.add_argument("-n", type=int, default=200)
args = parser.parse_args()

np.random.seed(args.seed)

env = GaussianEnvironment(1, 2, with_bias=True)
ls_param, ls_co = env.default_lengthscales()
param_init = env.default_init_param()
e_length = np.array([8.0, 8.0])

if not args.cache:
    X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
    metric = CompositeMetric.from_ls_list([ls_param, ls_co])

    config = ActiveSamplerConfig(n_mc_param_search=100, n_grid=20, integration_method="mc")
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)

    for i in tqdm.tqdm(range(args.n)):
        print(i)
        x = sampler.ask()
        sampler.tell(x, env.isInside(x))

    for i in tqdm.tqdm(range(30)):
        x = sampler.ask_additional()
        sampler.tell(x, env.isInside(x))

    with open("cache.pkl", "wb") as f:
        dill.dump(sampler, f)

with open("cache.pkl", "rb") as f:
    sampler: HolllessActiveSampler = dill.load(f)

best_param = sampler.best_param_so_far
fig, ax = plt.subplots()
# ax.plot(sampler.sampler_cache.best_volume_history)
sampler.fslset.show_sliced(best_param, list(range(len(best_param))), 30, (fig, ax))
env.visualize_optimal_sliced_region((fig, ax))
# axis
if env.with_bias:
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
else:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
plt.show()
