from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn


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
        return abs(np.cos(theta) - (-1)) < 0.04 and abs(theta_dot) < 0.1
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
        layers = []
        layers.append(nn.Linear(4, 12))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(12, 12))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(12, 1))
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
        residual_scaling: float = 1.0,
    ):
        self.base_controller = base_controller
        self.residual_net = residual_net
        self.residual_scaling = residual_scaling

    def __call__(self, state: np.ndarray) -> float:
        state_torch = th.from_numpy(state).float()
        action_residual = self.residual_net(state_torch).detach().cpu().numpy().item()
        action_base = self.base_controller(state)
        print(f"action_base: {action_base}, action_residual: {action_residual}")
        action_total = action_base + self.residual_scaling * action_residual
        return action_total


if __name__ == "__main__":
    net = ResidualPolicyNet()
    net.set_parameter(np.random.randn(net.dof) * 0.0)
    policy = Policy(EnergyShapingController(ModelParameter()), net)

    system = Cartpole(np.array([0.0, 0.0, 0.3, 0.0]))
    for _ in range(1000):
        state = system.state
        action = policy(state)
        system.step(action)
        if system.is_uplight():
            print("success")
            break
