import numpy as np

from .optimizer import BaseOptimizer

class SGD(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 1.0):
        self.params = params
        self.lr = lr 
        
        self.defaults = dict(lr=lr)
        
    def step(self, loss, grad):
        
        self.params -= self.lr * grad
            
        return loss, grad