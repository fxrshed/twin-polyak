import torch
import numpy as np
class BaseOracle(object):
    
    def func(self, x, data, target):
        raise NotImplementedError
    
    def grad(self, x, data, target):
        raise NotImplementedError
    
    def hess(self, x, data, target):
        raise NotImplementedError

class LogisticRegressionLoss(BaseOracle):
    
    def __init__(self, lmd: float = 0.0) -> None:
        self.lmd = lmd
   
    def func(self, x, data, target):
        return np.mean(np.log(1 + np.exp( - data@x * target ))) + self.lmd/2 * np.linalg.norm(x)**2
    
    def grad(self, x, data, target):
        r = np.exp( - data@x * target )
        ry = -r/(1+r) * target
        return (data.T @ ry )/data.shape[0]  + self.lmd * x
    
    def hess(self, x, data, target):
        r = np.exp( - data@x * target )
        rr= r/(1+r)**2
        return (data.T@np.diagflat(rr)@data) / data.shape[0] + self.lmd*np.eye(data.shape[1])
    
    def func_grad_acc(self, x, data, target):
        sparse_dot = data@x
        t = - sparse_dot * target
        f_val = np.mean(np.log(1 + np.exp( t ))) + self.lmd/2 * np.linalg.norm(x)**2
        
        r = np.exp(t)
        ry = -r/(1+r) * target
        grad_val = (data.T @ ry )/data.shape[0]  + self.lmd * x
        
        acc = (np.sign(sparse_dot) == target).sum() / target.shape[0]

        return f_val, grad_val, acc

class LeastSquaresLoss(BaseOracle):
    
    def func(self, w, data, target):
        return (0.5 / data.shape[0]) * np.linalg.norm((data@w) - target)**2
    
    def grad(self, w, data, target):
        return (data.T @ (data@w - target)) / data.shape[0]
    
    def hess(self, w, data, target):
        return (data.T @ data) / data.shape[0]
    
    
class CrossEntropyLoss(BaseOracle):

    def func(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    
    def grad(self, data, y_true, y_pred):
        return ( data.T.dot(y_pred - y_true) )  / data.shape[0] 
    
    def hess(self, x, data, target):
        raise NotImplementedError