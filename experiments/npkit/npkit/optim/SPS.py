import numpy as np

from .optimizer import BaseOptimizer


class SPS(BaseOptimizer):

    def __init__(self, params: np.ndarray,
                 c: float = 0.5,
                 eps: float = 1e-8,
                 eta_max: float = np.inf,
                 lr: float = 1.0):

        self.params = params
        self.lr = lr
        self.eps = eps
        self.c = c
        self.eta_max = eta_max

        self.defaults = dict(
            lr=lr,
            c=c,
            eps=eps,
            eta_max=eta_max
            )

    def step(self, loss, grad):

        self.lr = loss / (self.c * np.square(np.linalg.norm(grad)) + self.eps)
        self.lr = np.minimum(self.eta_max, self.lr)
        self.params -= self.lr * grad

        return loss, grad