import numpy as np
import tqdm

from frmax2.core import ActiveSamplerConfig, HolllessActiveSampler
from frmax2.environment import GaussianEnvironment
from frmax2.initialize import initialize
from frmax2.metric import CompositeMetric


def test_hollless_algorithm_1d():
    env = GaussianEnvironment(1, 1, with_bias=True)
    ls_param, ls_co = env.default_lengthscales()
    param_init = env.default_init_param()
    e_length = np.array([8.0])

    X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
    ls_co = np.array([0.3])
    metric = CompositeMetric.from_ls_list([ls_param, ls_co])

    config = ActiveSamplerConfig(n_mc_param_search=100, n_grid=30, box_cut=False, c_svm=100.0)
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)
    for _ in tqdm.tqdm(range(100)):
        x = sampler.ask()
        y = env.isInside(x)
        sampler.tell(x, y)
    volume_latest = sampler.sampler_cache.best_volume_history[-1]
    assert volume_latest > 1.8  # optimal is 2.0


def test_hollless_algorithm_2d():
    env = GaussianEnvironment(2, 2, with_bias=True)
    ls_param, ls_co = env.default_lengthscales()
    param_init = env.default_init_param()
    e_length = np.array([8.0, 8.0])

    X, Y, ls_co = initialize(lambda x: +1 if env.isInside(x) else -1, param_init, e_length, eps=0.2)
    metric = CompositeMetric.from_ls_list([ls_param, ls_co])

    config = ActiveSamplerConfig(n_mc_param_search=100, n_grid=20, integration_method="mc")
    sampler = HolllessActiveSampler(X, Y, metric, param_init, config)

    for i in tqdm.tqdm(range(400)):
        x = sampler.ask()
        sampler.tell(x, env.isInside(x))

    for i in tqdm.tqdm(range(30)):
        x = sampler.ask_additional()
        sampler.tell(x, env.isInside(x))

    best_param = sampler.sampler_cache.best_param_history[-1]
    best_vol = sampler.compute_sliced_volume(best_param)
    assert best_vol > 3.0  # optimal is pi
