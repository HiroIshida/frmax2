import argparse

import matplotlib.pyplot as plt
import numpy as np
from environment import GaussianEnvironment
from frmax.initialize import initialize

from frmax2.core import ActiveSamplerConfig, HolllessActiveSampler, NaiveActiveSampler
from frmax2.metric import CompositeMetric

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="hollless")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", type=int, default=70)
args = parser.parse_args()

np.random.seed(args.seed)


env = GaussianEnvironment(1, 1)
ls_param, ls_co = env.default_lengthscales()
param_init = env.default_init_param()
e_length = np.array([8.0])

X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
ls_co = np.array([0.3])
metric = CompositeMetric.from_ls_list([ls_param, ls_co])

config = ActiveSamplerConfig()
if args.method == "hollless":
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)
else:
    sampler = NaiveActiveSampler(X, Y, metric, param_init, config)

sampler = NaiveActiveSampler(X, Y, metric, param_init, config)
for i in range(args.n):
    print(i)
    x = sampler.ask()
    sampler.tell(x, env.isInside(x), update_clf=False)

# plot
fig, ax = plt.subplots()
env.visualize_region(-2.5, 1.5, (fig, ax))

X_positive = sampler.X[sampler.Y]
X_negative = sampler.X[~sampler.Y]
ax.scatter(X_positive[:, 0], X_positive[:, 1], c="b")
ax.scatter(X_negative[:, 0], X_negative[:, 1], c="r")

xlin = np.linspace(-2.5, 1.5, 100)
ylin = np.linspace(-1.5, 1.5, 100)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)
pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
values = sampler.fslset.func(pts)
Z = values.reshape(Xgrid.shape)
ax.contour(Xgrid, Ygrid, Z, levels=[0], cmpa="jet", zorder=1)
ax.set_aspect("equal")

# min max ax
ax.set_xlim(-2.5, 1.5)
ax.set_ylim(-1.5, 1.5)
plt.show()
