import argparse

import matplotlib.pyplot as plt
import numpy as np
from environment import GaussianEnvironment
from frmax.initialize import initialize

from frmax2.core import ActiveSampler
from frmax2.metric import CompositeMetric
from frmax2.region import FactorizableSuperLevelSet

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", type=int, default=70)
args = parser.parse_args()

np.random.seed(args.seed)


env = GaussianEnvironment(1, 1)
ls_param, ls_co = env.default_lengthscales()
param_init = env.default_init_param()
e_length = np.array([8.0])

X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
metric = CompositeMetric.from_ls_list([ls_param, ls_co])
fslset = FactorizableSuperLevelSet.fit(X, Y, metric, 50, C=1e4)

# run
sampler = ActiveSampler(fslset, param_init)
for i in range(args.n):
    print(i)
    x = sampler.ask()
    sampler.tell(x, env.isInside(x))

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
values = sampler.fslset.slset.func(pts)
Z = values.reshape(Xgrid.shape)
ax.contour(Xgrid, Ygrid, Z, levels=[0], cmpa="jet", zorder=1)
ax.set_aspect("equal")
plt.show()
