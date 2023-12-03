from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler


@dataclass
class Bound:
    bmin: np.ndarray
    bmax: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.bmin) / (self.bmax - self.bmin)

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.bmax - self.bmin) + self.bmin

    def to_torch_bounds(self) -> torch.Tensor:
        return torch.Tensor([self.bmin, self.bmax])

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.bmin, self.bmax)


class SaasBoOptimzer:
    model: SaasFullyBayesianSingleTaskGP
    bound: Bound
    X: List[np.ndarray]
    y: List[float]

    def __init__(self, X: List[np.ndarray], y: List[float], bounds: Bound):
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.bound = bounds
        self.fit_gp(X, y)
        self.X = X
        self.y = y

    def fit_gp(self, X: List[np.ndarray], y: List[float]):
        X_normalized = [self.bound.normalize(x) for x in X]
        X_ = torch.Tensor(np.vstack(X_normalized))
        Y_ = torch.Tensor(y).unsqueeze(-1)
        self.model = SaasFullyBayesianSingleTaskGP(
            X_, Y_, train_Yvar=torch.full_like(Y_, 1e-6), outcome_transform=Standardize(m=1)
        )
        n_scale = 1
        warmup_steps = 64 * n_scale
        num_samples = 32 * n_scale
        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            disable_progbar=False,
            jit_compile=True,
        )

    def ask(self) -> np.ndarray:
        assert self.model is not None
        y_max = max(self.y)
        acq_func = qExpectedImprovement(model=self.model, sampler=self.sampler, best_f=y_max)
        dim = self.bound.bmin.shape[0]
        bounds = torch.zeros((2, dim))
        bounds[1, :] = 1.0
        candidates, acq_values = optimize_acqf(
            acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=100
        )
        x_normalized = candidates.numpy().flatten()
        x = self.bound.de_normalize(x_normalized)
        return x

    def tell(self, x: np.ndarray, y: float):
        self.X.append(x)
        self.y.append(y)
        self.fit_gp(self.X, self.y)


if __name__ == "__main__":

    def gen_branin_high_dimensional(dim: int, random: bool = False):
        def branin_function(x, y):
            a = 1
            b = 5.1 / (4 * np.pi**2)
            c = 5 / np.pi
            r = 6
            s = 10
            t = 1 / (8 * np.pi)
            return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s

        if random:
            i, j = np.random.choice(dim, 2, replace=False)
        else:
            i, j = 0, 1

        def inner(x: np.ndarray):
            assert len(x) == dim
            return -branin_function(x[i], x[j])

        return inner

    n_dim = 25
    f_bench = gen_branin_high_dimensional(n_dim, random=False)
    bmin = -15 * np.ones(n_dim)
    bmax = 15 * np.ones(n_dim)
    bound = Bound(bmin, bmax)

    train_X = [bound.sample() for _ in range(20)]
    train_Y = [f_bench(x) for x in train_X]

    bo = SaasBoOptimzer(train_X, train_Y, bound)
    for _ in range(25):
        x = bo.ask()
        y = f_bench(x)
        bo.tell(x, y)
        print(f"x: {x}, y: {y}")
        print(f"best: {max(bo.y)}")
