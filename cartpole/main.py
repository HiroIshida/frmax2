import argparse
import logging
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import tqdm
from common import (
    Cartpole,
    Controller,
    EnergyShapingController,
    LQRController,
    ModelParameter,
)

from frmax2.core import ActiveSamplerConfig, DistributionGuidedSampler, SamplerCache
from frmax2.metric import CompositeMetric
from frmax2.utils import create_default_logger


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
    base_controller: Controller
    residual_net: ResidualPolicyNet
    residual_scaling: float

    def __init__(
        self,
        base_controller: Controller,
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
        # print(f"action_base: {action_base}, action_residual: {action_residual}")
        action_total = action_base + self.residual_scaling * action_residual
        return action_total


class ParameterizedController:
    def __init__(self, param: np.ndarray):
        energy_alpha = param[0]
        lqr_q = np.array(param[1:4])
        lqr_r = param[4]
        switch_coditing_theta = param[5]
        switch_coditing_theta_dot = param[6]
        nonlinear = EnergyShapingController(ModelParameter(), energy_alpha)
        linear = LQRController(ModelParameter(), lqr_q, lqr_r)

        def is_switchable(state: np.ndarray) -> bool:
            x, x_dot, theta, theta_dot = state
            return (
                abs(np.cos(theta) - (-1)) < switch_coditing_theta
                and abs(theta_dot) < switch_coditing_theta_dot
            )

        self.is_switchable = is_switchable
        self.nonlinear_mode = True
        self.nonlinear = nonlinear
        self.linear = linear

    def __call__(self, state: np.ndarray) -> float:
        if self.is_switchable(state):
            self.nonlinear_mode = False
        if self.nonlinear_mode:
            return self.nonlinear(state)
        else:
            return self.linear(state)[0]


class Environment:
    residual_net: ResidualPolicyNet
    param_dof: int

    def __init__(self, param_dof: int):
        self.residual_net = ResidualPolicyNet()
        self.param_dof = param_dof

    def rollout(self, x: np.ndarray) -> bool:
        # param_dof = self.residual_net.dof
        param_dof = self.param_dof
        param = x[:param_dof]
        error = x[param_dof:]
        res = self._rollout(param, error)
        print(f"param: {param}, error: {error}, res: {res}")
        return res

    def _rollout(self, param: np.ndarray, error: np.ndarray) -> bool:
        # self.residual_net.set_parameter(param)
        # base_controller = Controller(ModelParameter())
        # policy = Policy(base_controller, self.residual_net)
        policy = ParameterizedController(param)
        assert len(error) <= 3
        m = 1.0 + error[0]
        M = 1.0 + error[1] if len(error) > 1 else 1.0
        l = 1.0 + error[2] if len(error) > 2 else 1.0
        model_param = ModelParameter(m=m, M=M, l=l)
        system = Cartpole(np.array([0.0, 0.0, 0.1, 0.0]), model_param=model_param)
        for i in range(300):
            u = policy(system.state)
            system.step(u)
            x, _, _, _ = system.state
            if abs(x) > 10.0:
                return False
            if system.is_static():
                return True
        return False


if __name__ == "__main__":
    # argparse to select mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("-m", type=int, default=1, help="number of error dim")

    args = parser.parse_args()

    logger = create_default_logger(Path("/tmp/"), "train", logging.INFO)
    param_dof = 7

    file_path = Path(f"./sampler-cache-{args.m}.pkl")
    param_hand = np.array([0.1, 1.0, 1.0, 1.0, 0.5, 0.3, 0.3])

    if args.mode == "train":
        np.random.seed(1)
        env = Environment(param_dof)
        param_init = param_hand
        ls_param = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1])
        print(f"param_init: {param_init.shape}")

        X = []
        Y = []
        for i in tqdm.tqdm(range(100)):
            e = -np.ones(args.m) * 0.5 + np.random.rand(args.m)
            res = env._rollout(param_init, e)
            Y.append(res)
            X.append(np.hstack([param_init, e]))
        X = np.array(X)
        Y = np.array(Y)
        assert sum(Y) > 0
        logger.info(f"initial volume: {np.sum(Y) / len(Y)}")

        def situation_sampler() -> np.ndarray:
            w = 1.5
            e = np.random.rand(args.m) * w - 0.5 * w
            return np.array(e)

        if args.m == 1:
            n_mc_integral = 80
            learning_rate = 0.2
            r_exploration = 2.0
            ls_error = np.array([0.02] * args.m)
        elif args.m == 2:
            n_mc_integral = 120
            learning_rate = 0.05
            r_exploration = 2.0
            ls_error = np.array([0.05] * args.m)
        else:
            n_mc_integral = 300
            learning_rate = 0.05
            # r_exploration = 1.2
            r_exploration = 2.0
            ls_error = np.array([0.05] * args.m)
        metric = CompositeMetric.from_ls_list([ls_param, ls_error])
        config = ActiveSamplerConfig(
            n_mc_param_search=10,
            c_svm=1000,
            integration_method="mc",
            n_mc_integral=n_mc_integral,
            r_exploration=r_exploration,
            learning_rate=learning_rate,
            param_ls_reduction_rate=0.998,
        )

        def is_valid_param(param: np.ndarray) -> bool:
            (
                energy_alpha,
                lqr_q1,
                lqr_q2,
                lqr_q3,
                lqr_r,
                switch_coditing_theta,
                switch_coditing_theta_dot,
            ) = param
            if energy_alpha < 0.0:
                return False
            eps = 0.05
            if lqr_q1 < eps or lqr_q2 < eps or lqr_q3 < eps or lqr_r < eps:
                return False
            if switch_coditing_theta < 0.0 or switch_coditing_theta_dot < 0.0:
                return False
            return True

        sampler = DistributionGuidedSampler(
            X,
            Y,
            metric,
            param_init,
            situation_sampler=situation_sampler,
            config=config,
            is_valid_param=is_valid_param,
        )

        for i in tqdm.tqdm(range(args.n)):
            sampler.update_center()
            while True:
                x = sampler.ask()
                if x is not None:
                    break
            sampler.tell(x, env.rollout(x))
            if i % 10 == 0:
                # save sampler
                with file_path.open(mode="wb") as f:
                    dill.dump(sampler, f)
    elif args.mode == "test":
        # load sampler
        with file_path.open(mode="rb") as f:
            sampler: DistributionGuidedSampler = dill.load(f)
            sampler_cache: SamplerCache = sampler.sampler_cache
        import matplotlib.pyplot as plt

        print(sampler_cache.best_volume_history)
        print(sampler_cache.best_param_history[-1])
        plt.plot(sampler_cache.best_volume_history)
        plt.show()

        env = Environment(param_dof)
        param_init = sampler_cache.best_param_history[-1]
        volume = sampler_cache.best_volume_history[-1]
        print(f"expected volume: {volume}")

        # benchmark
        e_list = []
        bools = []
        bools_hand = []
        for i in tqdm.tqdm(range(100)):
            e = sampler.situation_sampler()
            res = env._rollout(param_init, e)
            res_hand = env._rollout(param_hand, e)
            e_list.append(e)
            bools.append(res)
            bools_hand.append(res_hand)
        print(f"hand volume: {np.sum(bools_hand) / len(bools_hand)}")
        print(f"actual volume: {np.sum(bools) / len(bools)}")

        if args.m == 1:
            es = np.linspace(min(e_list), max(e_list), 100)
            decision_values = [
                sampler.fslset.func(np.array([np.hstack([param_init, e])])) for e in es
            ]
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(e_list, bools, "o")
            ax1.plot(es, decision_values, "-")
            ax2.plot(e_list, bools_hand, "o")
            plt.show()
        elif args.m == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            E = np.array(e_list)
            bools = np.array(bools)
            bools_hand = np.array(bools_hand)
            ax1.scatter(E[bools, 1], E[bools, 0], c="blue")
            ax1.scatter(E[~bools, 1], E[~bools, 0], c="red")
            sampler.fslset.show_sliced(param_init, list(range(len(param_init))), 50, (fig, ax1))
            ax2.scatter(E[bools_hand, 1], E[bools_hand, 0], c="blue")
            ax2.scatter(E[~bools_hand, 1], E[~bools_hand, 0], c="red")
            plt.show()
        elif args.m == 3:
            print("do nothing")
        else:
            assert False
