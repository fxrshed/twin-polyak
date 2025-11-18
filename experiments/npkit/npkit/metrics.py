import numpy as np


class LogisticRegressionAccuracy:

    def __call__(self, logits, targets):
        preds = np.where(logits > 0, 1, -1)
        return np.mean(preds == targets)