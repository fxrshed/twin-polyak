import numpy as np

from .optimizer import BaseOptimizer

class DecSPS(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 c_0: float = 1.0, 
                 eps: float = 1e-8,
                 eta_max: float = np.inf,
                 f_star: float = 0.0):
        
        self.params = params
        self.lr = eta_max
        self.eps = eps
        self.c_0 = c_0
        self.c = c_0
        self.c_prev = self.c
        self.eta_max = eta_max
        self.f_star = f_star
        
        self.defaults = dict(
            c_0=c_0,
            eps=eps,
            eta_max=eta_max,
            f_star=f_star
            )
        
        self._step_t = 0
        
    def step(self, loss, grad):
        
        c_prev = self.c
        self.c = self.c_0 * np.sqrt(self._step_t + 1)
        self._step_t += 1
        
        polyak_lr = (loss - self.f_star) / (np.square(np.linalg.norm(grad)) + self.eps)
        self.lr = np.minimum(polyak_lr, c_prev * self.lr) / self.c

        self.params -= self.lr * grad
            
        return loss, grad