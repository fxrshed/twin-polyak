import numpy as np

from .optimizer import BaseOptimizer


class STP(BaseOptimizer):
    """Implements Stochastic Twin Polyak optimization method for linear models. 
    
    Args:
        params (np.ndarray): model parameters.
        beta (float, optional): momentum parameter. Defaults to 0.9.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Defaults to 1e-8.

    Example:
        >>> ...
        >>> optimizer_x = STP(model_x.params, **optimizer_hparams)
        >>> optimizer_y = STP(model_y.params, **optimizer_hparams)
        >>> ...
        >>> h_x = optimizer_x.momentum_step(loss_x, grad_x)
        >>> h_y = optimizer_y.momentum_step(loss_y, grad_y)
        >>> if h_x > h_y:
        >>>     optimizer_x.step(h_x - h_y)
        >>> else:
        >>>     optimizer_y.step(h_y - h_x)

    """
    
    def __init__(self, params: np.ndarray, 
                 beta: float = 0.9,
                 eps: float = 1e-8):
        
        self.params = params
        self.beta = beta
        self.eps = eps
        
        self.defaults = dict(
            beta=beta,
            eps=eps,
            )
        
        self.f_ma = 0.0
        self.g_ma = np.zeros_like(params)
        self.gamma_ma = 0.0

    def step(self, loss_diff):

        self.lr = loss_diff / (0.5 * np.square(np.linalg.norm(self.g_ma)) + self.eps) 
        self.params -= self.lr * self.g_ma
            
    def momentum_step(self, loss, grad):

        self.f_ma = self.beta * self.f_ma + (1 - self.beta) * loss
        self.gamma_ma = self.beta * self.gamma_ma + (1 - self.beta) * np.dot(grad, self.params)
        self.g_ma = self.beta * self.g_ma + (1 - self.beta) * grad
        h = self.f_ma + np.dot(self.g_ma, self.params) - self.gamma_ma

        return h

    def __str__(self):
        return 'STP'