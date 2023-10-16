from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from control import lqr


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
        return abs(np.cos(theta) - (-1)) < 0.3 and abs(theta_dot) < 0.3

    def is_static(self) -> bool:
        x, x_dot, theta, theta_dot = self.state
        return abs(np.cos(theta) - (-1)) < 0.01 and abs(theta_dot) < 0.01 and abs(x_dot) < 0.01


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


class LQRController:
    model_param: ModelParameter
    A: np.ndarray
    B: np.ndarray
    q_scale: np.ndarray
    r_scale: float

    def __init__(
        self,
        model_param: ModelParameter,
        q_scale: Union[float, np.ndarray] = 1.0,
        r_scale: float = 0.1,
    ):
        m, M, l, g = model_param.m, model_param.M, model_param.l, model_param.g

        def x_acc(theta, theta_dot, u) -> float:
            return (u + m * np.sin(theta) * (l * theta_dot**2 + g * np.cos(theta))) / (
                M + m * np.sin(theta) ** 2
            )

        def theta_acc(theta, theta_dot, u) -> float:
            x_acc = (u + m * np.sin(theta) * (l * theta_dot**2 + g * np.cos(theta))) / (
                M + m * np.sin(theta) ** 2
            )
            theta_acc = -1 * (np.cos(theta) * x_acc + g * np.sin(theta)) / l
            return theta_acc

        eps = 1e-6
        A_x = np.zeros((1, 2))
        A_x[0, 0] = (x_acc(np.pi + eps, 0.0, 0.0) - x_acc(np.pi, 0.0, 0.0)) / eps
        A_x[0, 1] = (x_acc(np.pi, eps, 0.0) - x_acc(np.pi, 0.0, 0.0)) / eps
        b_x = (x_acc(np.pi, 0.0, 0.0 + eps) - x_acc(np.pi, 0.0, 0.0)) / eps

        A_theta = np.zeros((2, 2))
        A_theta[0, 0] = 0
        A_theta[0, 1] = 1.0
        A_theta[1, 0] = (theta_acc(np.pi + eps, 0.0, 0.0) - theta_acc(np.pi, 0.0, 0.0)) / eps
        A_theta[1, 1] = (theta_acc(np.pi, eps, 0.0) - theta_acc(np.pi, 0.0, 0.0)) / eps
        B_theta = np.zeros((2, 1))
        B_theta[1, 0] = (theta_acc(np.pi, 0.0, 0.0 + eps) - theta_acc(np.pi, 0.0, 0.0)) / eps

        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        A[:2, :2] = A_theta
        A[2, :2] = A_x
        B[:2] = B_theta
        B[2] = b_x

        if isinstance(q_scale, float):
            q_scale = np.ones(3) * q_scale

        self.A = A
        self.B = B
        self.q_scale = q_scale
        self.r_scale = r_scale

    def __call__(self, state: np.ndarray):
        x, x_dot, theta, theta_dot = state
        Q = np.diag(self.q_scale)
        R = np.eye(1) * self.r_scale
        K, _, _ = lqr(self.A, self.B, Q, R)
        target_cand = [-3 * np.pi, -np.pi, np.pi, 3 * np.pi]
        target = min(target_cand, key=lambda x: abs(x - theta))
        u = -K @ np.array([theta - target, theta_dot, x_dot])
        return u


class Controller:
    nonlinear_controller: EnergyShapingController
    linear_controller: LQRController
    is_switchable: Callable[[np.ndarray], bool]
    nonlinear_mode: bool

    def __init__(
        self,
        model_param: ModelParameter,
        is_switchable: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        if is_switchable is None:

            def tmp(state: np.ndarray) -> bool:
                x, x_dot, theta, theta_dot = state
                return abs(np.cos(theta) - (-1)) < 0.3 and abs(theta_dot) < 0.3

            is_switchable = tmp
        self.model_param = model_param
        self.nonlinear_controller = EnergyShapingController(model_param)
        self.linear_controller = LQRController(model_param)
        self.is_switchable = is_switchable
        self.nonlinear_mode = True

    def __call__(self, state: np.ndarray) -> float:
        if self.is_switchable(state):
            self.nonlinear_mode = False
        if self.nonlinear_mode:
            return self.nonlinear_controller(state)
        else:
            return self.linear_controller(state)[0]
