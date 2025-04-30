from typing import Tuple
from collections import defaultdict

import numpy as np

from loss_functions import LogisticRegressionLoss

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
        
        self.lr = loss / ( self.c * np.square(np.linalg.norm(grad)) + self.eps )
        self.lr = np.minimum(self.eta_max, self.lr)
        self.params -= self.lr * grad
            
        return loss, grad
    
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
            lr=1.0,
            c_0=c_0,
            c=c_0,
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
    
def twin_polyak(dataset: list[np.ndarray], 
               batch_size: int, 
               n_epochs: int,
               eps: float = 0.0,
               seed: int = 0,
               ) -> dict: 
    
    train_data, train_target, test_data, test_target = dataset

    # parameters
    if batch_size == train_data.shape[0]:
        np.random.seed(seed)
    else:
        np.random.seed(0)
    params_x = np.random.randn(train_data.shape[1])
    params_y = np.random.randn(train_data.shape[1])

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=0.0)
    
    # logging 
    history = defaultdict(list)

    # Train Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, train_data, train_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["train/loss"].append(loss)
    history["train/acc"].append(acc)
    history["train/grad_norm_sq"].append(g_norm_sq)
    
    # Test Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, test_data, test_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["test/loss"].append(loss)
    history["test/acc"].append(acc)
    history["test/grad_norm_sq"].append(g_norm_sq)
    
    np.random.seed(seed)
    indices = np.arange(train_data.shape[0])
    
    for epoch in range(n_epochs):
    
        # Training 
        if batch_size != train_data.shape[0]:
            np.random.shuffle(indices)

        for idx in range(train_data.shape[0]//batch_size):
            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
            
            loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, batch_data, batch_target)
            loss_y, grad_y, acc_y  = loss_function.func_grad_acc(params_y, batch_data, batch_target)
    
            lr_x = np.minimum(( (loss_x - loss_y) / (0.5 * np.linalg.norm(grad_x)**2 + eps) ), np.inf) 
            lr_y = np.minimum(( (loss_y - loss_x) / (0.5 * np.linalg.norm(grad_y)**2 + eps) ), np.inf) 

            # Optimization step
            if loss_x > loss_y:
                params_x -= lr_x * grad_x
                lr = lr_x
            else:
                params_y -= lr_y * grad_y
                lr = lr_y         
            
            history["lr_x"].append(np.abs(lr_x))
            history["lr_y"].append(np.abs(lr_y))
            history["lr"].append(lr)
            
            
        # Train Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, train_data, train_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, train_data, train_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["train/loss"].append(loss)
        history["train/acc"].append(acc)
        history["train/grad_norm_sq"].append(g_norm_sq)
            
        # Test Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, test_data, test_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, test_data, test_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["test/loss"].append(loss)
        history["test/acc"].append(acc)
        history["test/grad_norm_sq"].append(g_norm_sq)

    return history


def twin_polyak_ma(dataset: list[np.ndarray], 
               batch_size: int, 
               n_epochs: int,
               seed: int = 0,
               beta: float = 0.9,
               eps: float = 0.0,
               ) -> dict: 
    
    np.random.seed(seed)

    train_data, train_target, test_data, test_target = dataset

    # parameters
    params_x = np.random.randn(train_data.shape[1])
    params_y = np.random.randn(train_data.shape[1])

    # oracle 
    loss_function = LogisticRegressionLoss(lmd=0.0)
    
    # logging 
    history = defaultdict(list)

    indices = np.arange(train_data.shape[0])
    
    # Train Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, train_data, train_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["train/loss"].append(loss)
    history["train/acc"].append(acc)
    history["train/grad_norm_sq"].append(g_norm_sq)
    
    # Test Evaluation 
    loss, grad, acc = loss_function.func_grad_acc(params_x, test_data, test_target)
    g_norm_sq = np.linalg.norm(grad)**2
    history["test/loss"].append(loss)
    history["test/acc"].append(acc)
    history["test/grad_norm_sq"].append(g_norm_sq)
    
    fm_x = 0.0
    gm_x = np.zeros_like(params_x)
    gamma_x = 0.0
    
    fm_y = 0.0
    gm_y = np.zeros_like(params_y)
    gamma_y = 0.0
    
    # batch_size = int(train_data.shape[0] * 0.9)
    
    step_t = 0
    
    for epoch in range(n_epochs):
    
        # Training 
        if batch_size != train_data.shape[0]:
            np.random.shuffle(indices)

        for idx in range(train_data.shape[0]//batch_size):
            batch_indices = indices[idx*batch_size:(idx+1)*batch_size]
            
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices] 
            
            loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, batch_data, batch_target)
            loss_y, grad_y, acc_y  = loss_function.func_grad_acc(params_y, batch_data, batch_target)

            fm_x = beta * fm_x + (1 - beta) * loss_x
            gamma_x = beta * gamma_x + (1 - beta) * np.dot(grad_x, params_x)
            gm_x = beta * gm_x + (1 - beta) * grad_x
            h_x = fm_x + (np.dot(gm_x, params_x) - gamma_x)
            
            fm_y = beta * fm_y + (1 - beta) * loss_y
            gamma_y = beta * gamma_y + (1 - beta) * np.dot(grad_y, params_y)
            gm_y = beta * gm_y + (1 - beta) * grad_y
            h_y = fm_y + np.dot(gm_y, params_y) - gamma_y

            if h_x > h_y:
                diff = h_x - h_y
                lr_x = diff / (0.5 * np.square(np.linalg.norm(gm_x)) + eps) 
                params_x -= lr_x * gm_x
                lr = lr_x
                history["train/batch/loss"].append(loss_y)
                history["train/batch/grad_norm_sq"].append(np.linalg.norm(grad_y)**2)
            else:
                diff = h_y - h_x
                lr_y = diff / (0.5 * np.square(np.linalg.norm(gm_y)) + eps)
                params_y -= lr_y * gm_y
                lr = lr_y
                history["train/batch/loss"].append(loss_x)
                history["train/batch/grad_norm_sq"].append(np.linalg.norm(grad_x)**2)
            
            history["lr"].append(lr)
            
        # Train Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, train_data, train_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, train_data, train_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["train/loss"].append(loss)
        history["train/acc"].append(acc)
        history["train/grad_norm_sq"].append(g_norm_sq)
            
        # Test Evaluation 
        loss_x, grad_x, acc_x = loss_function.func_grad_acc(params_x, test_data, test_target)
        loss_y, grad_y, acc_y = loss_function.func_grad_acc(params_y, test_data, test_target)
        
        if loss_x < loss_y:
            loss, grad, acc, g_norm_sq = loss_x, grad_x, acc_x, np.linalg.norm(grad_x)**2
        else:
            loss, grad, acc, g_norm_sq = loss_y, grad_y, acc_y, np.linalg.norm(grad_y)**2
            
        history["test/loss"].append(loss)
        history["test/acc"].append(acc)
        history["test/grad_norm_sq"].append(g_norm_sq)

    return history




class SPS_MA(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 c: float = 0.5, 
                 eps: float = 1e-8,
                 eta_max: float = np.inf,
                 betas: tuple[float, float] = (0.9, 0.9)):
        
        self.params = params
        self.lr = 1.0
        self.eps = eps
        self.c = c
        self.betas = betas
        self.eta_max = eta_max
        
        self.defaults = dict(
            lr=1.0,
            c=c,
            eps=eps,
            )
        
        self._step_t = 0
        
        self.f_ma = 0.0
        self.g_ma = np.zeros_like(self.params)
        
    def step(self, loss, grad):
        
        self._step_t += 1
        
        self.f_ma = self.betas[0] * self.f_ma + (1 - self.betas[0]) * loss
        self.g_ma = self.betas[1] * self.g_ma + (1 - self.betas[1]) * grad
        
        self.lr = self.f_ma / ( self.c * np.square(np.linalg.norm(self.g_ma)) + self.eps )
        self.lr = np.minimum(self.eta_max, self.lr)
        
        self.params -= self.lr * self.g_ma
            
        return loss, grad    
    
    
class SGD_MA(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr: float = 0.1,
                 beta: float = 0.0):
        
        self.params = params
        self.lr = lr  
        self.beta = beta
        
        self.defaults = dict(
            lr=1.0,
            beta=beta,
            )
        
        self.g_ma = np.zeros_like(self.params)
        
    def step(self, loss, grad):
        
        self.g_ma = self.beta * self.g_ma + (1 - self.beta) * grad
        self.params -= self.lr * self.g_ma
            
        return loss, grad    


class SGD_Momo(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, 
                 lr: float = 1.0,
                 beta: float = 0.9,
                 eps: float = 0.0):
        
        self.params = params
        self.lr_min = lr  
        self.beta = beta
        self.eps = eps
        
        self.defaults = dict(
            lr=lr,
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
        self.lr = np.minimum(self.lr_min, (np.maximum(h, 0) / (np.square(np.linalg.norm(self.g_ma)) + self.eps) )  )

        self.params -= self.lr * self.g_ma
        
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
