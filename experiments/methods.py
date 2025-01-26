from typing import Tuple

import numpy as np

class BaseOptimizer(object):
    
    def step(self, loss=None, grad=None):
        raise NotImplementedError

class SGD(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 1.0):
        self.params = params
        self.lr = lr 
        
        self.defaults = dict(lr=lr)
        
    def step(self, loss, grad):
        
        self.params -= self.lr * grad
            
        return loss, grad
    
    
class SPS(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 c: float = 0.5, 
                 eps: float = 1e-8,
                 eta_max: float = np.inf):
        
        self.params = params
        self.lr = 1.0
        self.eps = eps
        self.c = c
        self.eta_max = eta_max
        
        self.defaults = dict(
            lr=1.0,
            c=c,
            eps=eps,
            )
        
    def step(self, loss, grad):
        
        self.lr = loss / ( self.c * np.linalg.norm(grad)**2 + self.eps )
        self.lr = np.minimum(self.eta_max, self.lr)
        self.params -= self.lr * grad
            
        return loss, grad

class Adagrad(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 0.01, eps: float = 1e-10):
        self.params = params
        self.lr = lr 
        self.eps = eps
        
        self.sum = np.zeros_like(params)
        
        self.defaults = dict(lr=lr, eps=eps)
        
    def step(self, loss, grad):
        self.sum += np.square(grad)
        self.params -= self.lr * (grad / (np.sqrt(self.sum) + self.eps))
            
        return loss, grad
    
    
class Adam(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
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
