import scipy 
import numpy as np

class BaseOracle(object):
    
    def func(self, *args, **kwargs):
        raise NotImplementedError
    
    def grad(self, *args, **kwargs):
        raise NotImplementedError
    
    def hess(self, *args, **kwargs):
        raise NotImplementedError

class LogisticRegressionLoss(BaseOracle):
    
    def __init__(self, lmd: float = 0.0) -> None:
        """Initialize the logistic regression loss function with optional L2 regularization.

        The regularized logistic loss function is defined as:
        .. math::

        f(w) = \\frac{1}{n} \\sum_{i=1}^n \\log\\bigl(1 + e^{-y_i \\langle x_i, w \\rangle}\\bigr)
        + \\frac{\\lambda}{2} \\|w\\|^2

    where:
        - :math:`x_i` is the feature vector of sample i.
        - :math:`y_i` is the label of sample i, typically in {-1, 1}.
        - :math:`w` is the weight vector.
        - :math:`\\lambda` is the regularization parameter.


        Args:
            lmd (float, optional): regularization parameter. Defaults to 0.0, meaning no regularization.
        """ 
        self.lmd = lmd
   
    def loss(self, logits, y, w = None):
        """Compute logist loss

        Args:
            logits (ndarray): Linear model outputs (before applying sigmoid), of shape (n_samples, ), i.e. X @ w.
            y (ndarray): Target labels of shape (n_samples, ), expected to be in {-1, 1}.
            w (ndarray, optional): Weight vector of shape (n_features, ), must be provided if regularization parameter is > 0. Defaults to None.

        Returns:
            float: The scalar value of the logistic loss with regularization term if it is > 0
        """
        reg_term = 0.0
        if self.lmd > 0.0:
            if w is not None:
                reg_term = 0.5 * self.lmd * np.linalg.norm(w)**2
        loss_val = np.mean(np.logaddexp(0, -y * logits))
        return loss_val + reg_term
        
    def grad(self, logits, X, y, w = None) -> np.array:
        """Compute the gradient of the logistic loss (with optional regularization term) with respect to the weights.

        Args:
            logits (ndarray): Linear model outputs (before applying sigmoid), of shape (n_samples, ), i.e. X @ w.
            X (ndarray): Input data matrix of shape (n_sample, n_features).
            y (ndarray): Target labels of shape (n_samples, ), expected to be in {-1, 1}.
            w (ndarray, optional): Weight vector of shape (n_features, ), must be provided if regularization parameter is > 0. Defaults to None.

        Returns:
            np.array: Gradient vector of shape (n_features, ) with respect to the weights.
        """
        s = self._sigmoid(-y * logits)
        
        reg_term = 0.0
        if self.lmd > 0.0:
            if w is not None:
                reg_term = self.lmd * w
            else:
                raise ValueError("Provide model parameters `w` for computing regularization term.")

        grad_val = (X.T @ (-y * s)) / X.shape[0]
        return grad_val + reg_term

    def hess(self, logits, X, y):
        """Compute the Hessian matrix of the logistic loss.

        This function computes the second derivative (Hessian) of the logistic loss function with respect the model parameters, 
        optionally incorporating L2 regularization.

        It assumes:
        - The model logits are precomputed as logits = X @ w.
        - Targets are binary {-1, 1}.add

        The Hessian has the form:
            H = (X^T * S * X) / n + lambda * I
        where:
            - S is a diagonal matrix with elements sigma(x_i) * (1 - sigma(x_i)),
            where sigma is the sigmoid function.
            - lambda is the L2 regularization parameter. 

        Args:
            logits (ndarray): Linear model outputs (before applying sigmoid), of shape (n_samples, ), i.e. X @ w.
            X (ndarray): Input data matrix of shape (n_sample, n_features).
            y (ndarray): Target labels of shape (n_samples, ), expected to be in {-1, 1}.

        Returns:
            ndarray: The Hessian matrix of shape (n_features, n_features).
        """
        s = self._sigmoid(-y * logits)
        d = s * (1 - s)
        
        if isinstance(X, scipy.sparse.csr_matrix):
            D = scipy.sparse.diags(d)
            X_weighted = D.dot(X)
            H = X.T.dot(X_weighted) / X.shape[0] 
        else:
            d = d[:, np.newaxis]
            X_weighted = X * d
            H = (X.T @ X_weighted) / X.shape[0] 
        
        reg_term = self.lmd * np.eye(X.shape[1])
        
        return H + reg_term
    
    def _sigmoid(self, z):
        """Compute the sigmoid function in a numerically stable way.

        The sigmoid function is defined as:
            sigmoid(z) = 1 / (1 + exp(-z))

        This implementation avoids numerical overflow/underflow by using
        a piecewise formulation.    

        Args:
            z (ndarray or float): Input value(s), can be a scalar or numpy array.

        Returns:
            ndarray or float: Sigmoid of the input, with the same shape as `z`.
        """
        return np.where(
                z >= 0,
                1 / (1 + np.exp(-z)),
                np.exp(z) / (1 + np.exp(z))
            )
    



class LegacyLogisticRegressionLoss(BaseOracle):
    
    def __init__(self, lmd: float = 0.0) -> None:
        self.lmd = lmd
   
    def loss(self, x, data, target):
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