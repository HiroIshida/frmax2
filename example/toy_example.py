import matplotlib.pyplot as plt
import numpy as np
from environment import GaussianEnvironment
from frmax.initialize import initialize

from frmax2.core import ActiveSampler
from frmax2.metric import CompositeMetric
from frmax2.region import FactorizableSuperLevelSet

env = GaussianEnvironment(1, 1)
ls_param, ls_co = env.default_lengthscales()
param_init = env.default_init_param()
e_length = np.array([8.0])

X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
metric = CompositeMetric.from_ls_list([ls_param, ls_co])
fslset = FactorizableSuperLevelSet.fit(X, Y, metric, 50)

sampler = ActiveSampler(fslset, param_init)
for i in range(50):
    print(i)
    x = sampler.ask()
    sampler.tell(x, env.isInside(x), False)

fig, ax = plt.subplots()
ax.scatter(sampler.X[:, 0], sampler.X[:, 1])
plt.show()
