from .optimizer import BaseOptimizer

import numpy as np

class SLS(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr_max: float = 1.0, 
                 gamma: float = 0.9, 
                 c: float = 0.1, 
                 verbose: bool = False):
        
        self.params = params
        self.gamma = gamma
        self.c = c # Sufficient decrease (Armijo) condition constant 
        self.lr_max = lr_max
        self.lr = lr_max
        self.verbose = verbose
        
        self.defaults = dict(
            lr_max=lr_max,
            gamma=gamma,
            c=c,
            verbose=verbose
            )
        
        self.adaptive_search_max_iter = 100
        
    def step(self, loss, grad, closure) -> np.ndarray:

        d = -1.0 * grad
        self.lr = self.lr_max

        for j in range(1, self.adaptive_search_max_iter + 1):
            if j == self.adaptive_search_max_iter:
                if self.verbose:
                    print(('Warning: adaptive_iterations_exceeded'), flush=True)
                break
            
            new_params = self.params + self.lr * d
            new_loss = closure(new_params)
            
            sufficient_decrease = new_loss <= loss + self.lr * self.c * d.dot(grad)
            
            if sufficient_decrease:
                if self.verbose:
                    print(f"Backtracking took {j} steps: lr={self.lr}")
                break
            
            self.lr = self.lr * self.gamma
            
        # Update the parameters
        self.params += self.lr * d

        return loss, grad