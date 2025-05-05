import torch
from torch import Tensor
from typing import Any, Dict, Iterable, List

class TwinPolyak(torch.optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], 
                 eps: float = 1e-8) -> None:
        
        self.step_size: float = 1.0
        
        defaults = dict(
            eps=eps,
            )

        super().__init__(params, defaults)
        
    def step(self, loss_diff: float):

        for group in self.param_groups:
            
            eps: float = group["eps"]
            
            grad_norm_squared: float = 0.0
            for p in group["params"]:
                grad_norm_squared += p.grad.data.mul(p.grad.data).sum()
            
            self.step_size = loss_diff / (0.5 * grad_norm_squared + eps)

            # Update parameters
            with torch.no_grad():
                for p in group["params"]:
                    p.sub_(p.grad.data, alpha=self.step_size)

class DecSPS(torch.optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], 
                 c_0: float = 1.0,
                 eta_max: float = 10.0,
                 f_star: float = 0.0,
                 eps: float = 1e-8) -> None:
        
        self.step_size: float = eta_max
        self.c_0: float = c_0
        self.c: float = c_0
        self.c_prev: float = c_0
        self.eta_max: float = eta_max
        self.f_star: float = f_star
    
        self._step_t: Tensor = torch.tensor(0)
        
        defaults = dict(
            c_0=c_0,
            c=c_0,
            eta_max=eta_max,
            eps=eps,
            )

        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            raise ValueError("Closure must be provided for loss evaluation.")
                
        for group in self.param_groups:
            
            eps: float = group["eps"]
            
            c_prev: Tensor = torch.tensor(self.c)
            self.c = self.c_0 * torch.sqrt(self._step_t + 1)
            self._step_t += 1
            
            grad_norm_squared: float = 0.0
            for p in group["params"]:
                grad_norm_squared += p.grad.data.mul(p.grad.data).sum()
                
            polyak_lr = (loss - self.f_star) / (grad_norm_squared + eps)
            self.step_size = (torch.minimum(polyak_lr, c_prev * self.step_size) / self.c).item()

            # Update parameters
            with torch.no_grad():
                for p in group["params"]:
                    p.sub_(p.grad.data, alpha=self.step_size)
                    
        return loss