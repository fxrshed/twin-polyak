import pytest 

import numpy as np
import sklearn.datasets

from npkit.loss import LogisticRegressionLoss, LegacyLogisticRegressionLoss

loss = LogisticRegressionLoss()
loss_legacy = LegacyLogisticRegressionLoss()

@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4, 5])
def test_logistic_loss(seed):
    
    np.random.seed(seed)
    n = 100
    d = 10

    data, target = sklearn.datasets.make_classification(n_samples=n, n_features=d, n_redundant=0, n_clusters_per_class=1, class_sep=3.0, random_state=seed)

    w = np.random.randn(data.shape[1])

    logits = data @ w

    np.testing.assert_almost_equal(loss.loss(logits, target, w), loss_legacy.loss(w, data, target))
    np.testing.assert_allclose(loss.grad(logits, data, target), loss_legacy.grad(w, data, target))
    np.testing.assert_allclose(loss.hess(logits, data, target), loss_legacy.hess(w, data, target))
