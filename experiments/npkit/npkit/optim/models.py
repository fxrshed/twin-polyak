import numpy as np


class NumpyLinearModel:
    def __init__(self, input_dim: int):
        self.params = np.random.randn(input_dim)

    def __call__(self, X):
        return X @ self.params

    def parameters(self):
        return self.params
