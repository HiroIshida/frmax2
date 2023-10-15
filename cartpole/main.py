import argparse
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import tqdm

from frmax2.core import ActiveSamplerConfig, DistributionGuidedSampler, SamplerCache
from frmax2.initialize import initialize
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger


@dataclass
class ModelParameter:
    m: float = 1.0
    M: float = 1.0
    l: float = 1.0
    g: float = 9.8

    @classmethod
    def create_random(cls) -> "ModelParameter":
        return cls(
            m=1.0 + np.random.randn() * 0.1,
            M=1.0 + np.random.randn() * 0.1,
            l=1.0 + np.random.randn() * 0.1,
            g=9.8,
        )


class Cartpole:
    state: np.ndarray
    model_param: ModelParameter
    history: List[np.ndarray]

    def __init__(self, state: np.ndarray, model_param: ModelParameter = ModelParameter()):
        self.state = state
        self.model_param = model_param
        self.history = []

    def step(self, f: float, dt: float = 0.05):
        x, x_dot, theta, theta_dot = self.state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        x_acc = (f + m * np.sin(theta) * (l * theta_dot**2 + g * np.cos(theta))) / (
            M + m * np.sin(theta) ** 2
        )
        theta_acc = -1 * (np.cos(theta) * x_acc + g * np.sin(theta)) / l
        x_dot += x_acc * dt
        theta_dot += theta_acc * dt
        x += x_dot * dt
        theta += theta_dot * dt
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.history.append(self.state)

    def is_uplight(self) -> bool:
        x, x_dot, theta, theta_dot = self.state
        return abs(np.cos(theta) - (-1)) < 0.04 and abs(theta_dot) < 0.05
        # return abs(np.cos(theta) - (-1)) < 0.01 and abs(theta_dot) < 0.01

    def render(self, ax):
        x, x_dot, theta, theta_dot = self.state
        l = self.model_param.l
        cart = plt.Circle((x, 0), 0.2, color="black")
        ax.add_patch(cart)
        pole = plt.Line2D((x, x + l * np.sin(theta)), (0, -l * np.cos(theta)), color="black")
        ax.add_line(pole)


class CartpoleVisualizer:
    def __init__(self, model_param: ModelParameter = ModelParameter()):
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.initialized = False
        self.model_param = model_param
        cart_circle = plt.Circle((0, 0), 0.2, color="black")
        self.cart = ax.add_patch(cart_circle)
        pole_line = plt.Line2D((0, 0), (0, -model_param.l), color="black")
        self.pole = ax.add_line(pole_line)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")

    def render(self, state):
        x, x_dot, theta, theta_dot = state
        l = self.model_param.l

        x = 0.0
        self.cart.set_center = (x, 0)
        self.pole.set_data((x, x + l * np.sin(theta)), (0, -l * np.cos(theta)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# Chung, Chung Choo, and John Hauser. "Nonlinear control of a swinging pendulum." automatica 31.6 (1995): 851-862.
class EnergyShapingController:
    model_param: ModelParameter
    alpha: float

    def __init__(self, model_param: ModelParameter, alpha: float = 0.1):
        self.model_param = model_param
        self.alpha = alpha

    def __call__(self, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        s, c = np.sin(theta), np.cos(theta)
        E_swing = 0.5 * m * l**2 * theta_dot**2 + m * g * l * (1 - c)
        E_swing_d = 2 * m * g * l
        u = self.alpha * theta_dot * c * (E_swing - E_swing_d)
        f = (M + m * s**2) * u - (m * l * s * theta_dot**2 + m * g * s * c)
        return f


class ResidualPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        n = 6
        # n = 12
        layers = []
        layers.append(nn.Linear(4, n))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n, n))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(n, 1))
        self.layers = nn.Sequential(*layers)

    @property
    def dof(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def set_parameter(self, set_param: np.ndarray):
        assert len(set_param) == self.dof
        head = 0
        for name, param in self.named_parameters():
            n = np.prod(param.shape)
            tail = head + n
            param_partial = set_param[head:tail].reshape(tuple(param.shape))
            param.data = param.new_tensor(param_partial)
            head += n

    def get_parameter(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])

    def forward(self, x):
        return self.layers(x)


class Policy:
    base_controller: EnergyShapingController
    residual_net: ResidualPolicyNet
    residual_scaling: float

    def __init__(
        self,
        base_controller: EnergyShapingController,
        residual_net: ResidualPolicyNet,
        residual_scaling: float = 10.0,
    ):
        self.base_controller = base_controller
        self.residual_net = residual_net
        self.residual_scaling = residual_scaling

    def __call__(self, state: np.ndarray) -> float:
        state_torch = th.from_numpy(state).float()
        action_residual = self.residual_net(state_torch).detach().cpu().numpy().item()
        action_base = self.base_controller(state)
        # print(f"action_base: {action_base}, action_residual: {action_residual}")
        action_total = action_base + self.residual_scaling * action_residual
        return action_total


class Environment:
    residual_net: ResidualPolicyNet

    def __init__(self):
        self.residual_net = ResidualPolicyNet()

    def rollout(self, x: np.ndarray) -> bool:
        param_dof = self.residual_net.dof
        param = x[:param_dof]
        error = x[param_dof:]
        res = self._rollout(param, error)
        print(f"param: {param.shape}, error: {error}, res: {res}")
        return res

    def _rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        self.residual_net.set_parameter(param)
        policy = Policy(EnergyShapingController(ModelParameter()), self.residual_net)
        assert len(error) == 1
        m = 1.0 + error[0]
        model_param = ModelParameter(m=m)
        system = Cartpole(np.zeros(4), model_param=model_param)
        for i in range(1000):
            state = system.state
            action = policy(state)
            system.step(action)
            if system.is_uplight():
                return True
        return False


if __name__ == "__main__":
    # argparse to select mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("-n", type=int, default=300)

    args = parser.parse_args()

    if args.mode == "train":
        np.random.seed(1)
        env = Environment()
        param_init = np.random.randn(env.residual_net.dof) * 0.01
        print(f"param_init: {param_init.shape}")
        e_list = np.linspace(-0.5, 0.5, 50)
        X = []
        Y = []
        for e in tqdm.tqdm(e_list):
            res = env._rollout(param_init, np.array([e]))
            Y.append(res)
            X.append(np.hstack([param_init, e]))
        X = np.array(X)
        Y = np.array(Y)
        assert sum(Y) > 0
        ls_param = np.array([0.3] * env.residual_net.dof)
        ls_error = np.array([0.01])
        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        config = ActiveSamplerConfig(
            n_mc_param_search=10,
            c_svm=100000,
            integration_method="mc",
            n_mc_integral=100,
            r_exploration=2.0,
            learning_rate=0.3,
        )

        def situation_sampler() -> np.ndarray:
            w = 1.5
            e = np.random.rand() * w - 0.5 * w
            return np.array([e])

        sampler = DistributionGuidedSampler(
            X, Y, metric, param_init, situation_sampler=situation_sampler, config=config
        )

        create_default_logger(Path("/tmp/"), "train", logging.INFO)

        for i in tqdm.tqdm(range(args.n)):
            print(i)
            x = sampler.ask()
            if x is None:
                continue
            sampler.tell(x, env.rollout(x))
            if i % 10 == 0:
                # save sampler
                file_path = Path("./sampler.pkl")
                with file_path.open(mode = "wb") as f:
                    pickle.dump(sampler.sampler_cache, f)
    elif args.mode == "test":
        # load sampler
        file_path = Path("./sampler.pkl")
        with file_path.open(mode = "rb") as f:
            sampler_cache: SamplerCache = pickle.load(f)
        import matplotlib.pyplot as plt
        print(sampler_cache.best_volume_history)
        print(sampler_cache.best_param_history[-1])
        plt.plot(sampler_cache.best_volume_history)
        plt.show()

        env = Environment()
        param_init = sampler_cache.best_param_history[-1]
        print(param_init)
        volume = sampler_cache.best_volume_history[-1]
        print(f"expected volume: {volume}")

        e_list = np.linspace(-0.75, 0.75, 100)
        bools = []
        for e in tqdm.tqdm(e_list):
            res = env._rollout(param_init, np.array([e]))
            bools.append(res)
        bools = np.array(bools)
        print(f"actual volume: {np.sum(bools) / len(bools)}")
        plt.plot(e_list, bools, "o")
        plt.show()
