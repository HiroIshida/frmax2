import pickle
from hashlib import md5

import numpy as np

from frmax2.core import (
    ActiveSamplerConfig,
    DGSamplerConfig,
    DistributionGuidedSampler,
    HolllessActiveSampler,
)
from frmax2.environment import GaussianEnvironment
from frmax2.initialize import initialize
from frmax2.metric import CompositeMetric


def test_regression_hollless():
    np.random.seed(0)
    env = GaussianEnvironment(1, 1, with_bias=True)
    ls_param, ls_co = env.default_lengthscales()
    param_init = env.default_init_param()
    e_length = np.array([8.0])

    X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
    ls_co = np.array([0.3])
    metric = CompositeMetric.from_ls_list([ls_param, ls_co])

    config = ActiveSamplerConfig(n_mc_param_search=100, n_grid=30, box_cut=False, c_svm=100.0)
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)

    for i in range(30):
        x = sampler.ask()
        sampler.tell(x, env.isInside(x))
    md5sum = md5(pickle.dumps((sampler.X, sampler.Y))).hexdigest()
    assert md5sum == "c8a63003bf5d154701d14d4902693f0f"


def test_regression_generalized():

    env = GaussianEnvironment(1, 1, with_bias=True)
    ls_param, ls_co = env.default_lengthscales()
    ls_param *= 2.0
    param_init = env.default_init_param()

    env = GaussianEnvironment(1, 1, with_bias=False, with_hollow=False)

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
        epsilon_exploration=0.2,
        param_ls_reduction_rate=0.995,
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

    for i in range(30):
        x = sampler.ask()
        sampler.tell(x, env.isInside(x))

    md5sum = md5(pickle.dumps((sampler.X, sampler.Y))).hexdigest()
    assert md5sum == "61e114aafcfbef58befb6b1ee78aae94"
