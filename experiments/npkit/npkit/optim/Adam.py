import numpy as np

from .optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    
    def __init__(self, params: np.ndarray,
                 lr: float = 0.001,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr 
        self.eps = eps
        self.betas = betas
        
        self.sum_m = np.zeros_like(params)
        self.sum_v = np.zeros_like(params)
        self._step_t: int = 0
        
        self.defaults = dict(lr=lr, eps=eps, betas=betas)
        
    def step(self, loss, grad):
        
        self._step_t += 1

        self.sum_m = self.betas[0] * self.sum_m + (1 - self.betas[0]) * grad
        self.sum_v = self.betas[1] * self.sum_v + (1 - self.betas[1]) * np.square(grad)
        m_hat = self.sum_m / (1 - self.betas[0]**self._step_t)
        v_hat = self.sum_v / (1 - self.betas[1]**self._step_t)
    
        self.params -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            
        return loss, grad