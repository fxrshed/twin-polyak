import numpy as np

from .optimizer import BaseOptimizer

class Adagrad(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr: float = 0.01, 
                 eps: float = 1e-10):
        self.params = params
        self.lr = lr 
        self.eps = eps
        
        self.sum = np.zeros_like(params)
        
        self.defaults = dict(lr=lr, eps=eps)
        
    def step(self, loss, grad):
        self.sum += np.square(grad)
        self.params -= self.lr * (grad / (np.sqrt(self.sum) + self.eps))
            
        return loss, grad