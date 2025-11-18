import numpy as np

from .optimizer import BaseOptimizer


class Momo(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr_min: float = 1.0,
                 beta: float = 0.9,
                 eps: float = 1e-8):
        
        self.params = params
        self.lr_min = lr_min
        self.beta = beta
        self.eps = eps
        
        self.defaults = dict(
            lr_min=lr_min,
            beta=beta,
            eps=eps,
            )
        
        self.f_ma = 0.0
        self.g_ma = np.zeros_like(params)
        self.gamma_ma = 0.0

    def step(self, loss, grad):
        
        self.f_ma = self.beta * self.f_ma + (1 - self.beta) * loss
        self.g_ma = self.beta * self.g_ma + (1 - self.beta) * grad
        self.gamma_ma = self.beta * self.gamma_ma + (1 - self.beta) * np.dot(grad, self.params)
        
        h = self.f_ma + np.dot(self.g_ma, self.params) - self.gamma_ma
        self.lr = np.minimum(self.lr_min, (np.maximum(h, 0) / (np.square(np.linalg.norm(self.g_ma)) + self.eps)))

        self.params -= self.lr * self.g_ma
        
        return loss, grad