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
    
class TwinPolyakMA(torch.optim.Optimizer):
    
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]],
                beta: float = 0.9,
                 eps: float = 1e-8) -> None:
        
        self.step_size: float = 1.0
        self.beta: float = beta
        
        defaults = dict(
            beta=beta,
            eps=eps,
            )
        
        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["gamma_avg"] = 0.0
                state["grad_avg"] = torch.zeros_like(p)
                
        self.loss_avg = 0.0
        
    def step(self, loss_diff: float, closure=None):

        for group in self.param_groups:
            
            eps: float = group["eps"]
            
            grad_avg_norm_squared: float = 0.0
            for p in group["params"]:
                grad_avg = self.state[p]["grad_avg"]
                grad_avg_norm_squared += grad_avg.data.mul(grad_avg.data).sum()
            
            self.step_size = loss_diff / (0.5 * grad_avg_norm_squared + eps)

            # Update parameters
            with torch.no_grad():
                for p in group["params"]:
                    grad_avg = self.state[p]["grad_avg"]
                    p.sub_(grad_avg, alpha=self.step_size)
                    
                    
    def momentum_step(self, loss=None):
        
        if loss is None:
            raise TypeError("Argument `loss` is required for gradient calculation.")
        else:
            loss.backward()
        
        h: float = 0.0
        
        self.loss_avg = self.beta * self.loss_avg + (1 - self.beta) * loss.item()
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                # state["loss_avg"] = beta * state["loss_avg"] + (1 - beta) * loss
                # grad = torch.autograd.grad(loss, p, retain_graph=True)[0]
                state["grad_avg"] = self.beta * state["grad_avg"] + (1 - self.beta) * grad
                grad_dot_p = torch.sum(p.data * grad)
                state["gamma_avg"] = self.beta * state["gamma_avg"] + (1 - self.beta) * grad_dot_p
                grad_avg_dot_p = torch.sum(p.data *  state["grad_avg"])
                # h += state["loss_avg"] + grad_avg_dot_p - state["gamma_avg"]
                h += grad_avg_dot_p - state["gamma_avg"]
        
        h += self.loss_avg
        
        return h.item() if isinstance(h, Tensor) else h
    
    
    
"""
Adapted from:
	Novik, Mykola: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/types.py
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]

"""
Implements the MoMo algorithm.

Authors: Fabian Schaipp, Ruben Ohana, Michael Eickenberg, Aaron Defazio, Robert Gower
"""
import torch
import warnings
from math import sqrt

class Momo(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1.0,
                 weight_decay: float=0,
                 beta: float=0.9,
                 lb: float=0,
                 bias_correction: bool=False,
                 use_fstar: bool=False) -> None:
        """
        MoMo optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1.
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        beta : float, optional
            Momentum parameter, should be in [0,1), by default 0.9.
        lb : float, optional
            Lower bound for loss. Zero is often a good guess.
            If no good estimate for the minimal loss value is available, you can set use_fstar=True.
            By default 0.
        bias_correction : bool, optional
            Which averaging scheme is used, see details in the paper. By default False.
        use_fstar : bool, optional
            Whether to use online estimation of loss lower bound. 
            Can be used if no good estimate is available, by default False.

        """
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        
        self.beta = beta
        self.lb = lb
        self._initial_lb = lb
        self.bias_correction = bias_correction
        self.use_fstar = use_fstar
        
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        
        return
        
    def step(self, closure: LossClosure=None, loss=None) -> OptFloat:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
        

        Returns
        -------
        (Stochastic) Loss function value.
        """

        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        self._number_steps += 1
        beta = self.beta  
        
        ###### Preliminaries
        if self._number_steps == 1:    
            if self.bias_correction:
                self.loss_avg = 0.
            else:
                self.loss_avg = loss.detach().clone()
        
        self.loss_avg = beta*self.loss_avg + (1-beta)*loss.detach()        

        if self.bias_correction:
            rho = 1-beta**self._number_steps # must be after incrementing k
        else:
            rho = 1
            
        _dot = 0.
        _gamma = 0.
        _norm = 0.
        
        ############################################################
        # Notation
        # d_k: p.grad_avg, gamma_k: _gamma, \bar f_k: self.loss_avg
        for group in self.param_groups:
            for p in group['params']:
                
                grad = p.grad.data.detach()
                state = self.state[p]

                # Initialize EMA
                if self._number_steps == 1:
                    if self.bias_correction:
                        state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                        state['grad_dot_w'] = torch.zeros(1).to(p.device)
                    else:
                        # Exponential moving average of gradients
                        state['grad_avg'] = grad.clone()
                        # Exponential moving average of inner product <grad, weight>
                        state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))
                        
                grad_avg, grad_dot_w = state['grad_avg'], state['grad_dot_w']

                grad_avg.mul_(beta).add_(grad, alpha=1-beta)
                grad_dot_w.mul_(beta).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta)

                _dot += torch.sum(torch.mul(p.data, grad_avg))
                _gamma += grad_dot_w
                _norm += torch.sum(torch.mul(grad_avg, grad_avg))

        #################   
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            if self.use_fstar:
                cap = ((1+lr*lmbda)*self.loss_avg + _dot - (1+lr*lmbda)*_gamma).item()
                # Reset
                if cap < (1+lr*lmbda)*rho*self.lb:
                    self.lb = cap/(2*(1+lr*lmbda)*rho) 
                    self.lb = max(self.lb, self._initial_lb) # safeguard

            ### Compute adaptive step size
            if lmbda > 0:
                nom = (1+lr*lmbda)*(self.loss_avg - rho*self.lb) + _dot - (1+lr*lmbda)*_gamma
                t1 = max(nom, 0.)/_norm
            else:
                t1 = max(self.loss_avg - rho*self.lb + _dot - _gamma, 0.)/_norm
            
            t1 = t1.item() # make scalar
            
            tau = min(lr/rho, t1) # step size

            ### Update lb estimator
            if self.use_fstar:
                h = (self.loss_avg  + _dot -  _gamma).item()
                self.lb = ((h - (1/2)*tau*_norm)/rho).item() 
                self.lb = max(self.lb, self._initial_lb) # safeguard

            ### Update params
            for p in group['params']:   
                state = self.state[p]
                grad_avg = state['grad_avg']          
                p.data.add_(other=grad_avg, alpha=-tau)
                
                if lmbda > 0:
                    p.data.div_(1+lr*lmbda)
                    
        ############################################################
        if self.use_fstar:
            self.state['fstar'] = self.lb
        
        # If you want to track the adaptive step size term, activate the following line.
        self.state['step_size_list'].append(tau)
        
        return loss
    
    
    
    
