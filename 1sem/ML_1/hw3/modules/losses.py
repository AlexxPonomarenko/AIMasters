import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        weights, bias = w[1:], w[0]
        y_predict = (X @ weights + bias) * -y
        return np.mean(np.logaddexp(np.zeros_like(y_predict), y_predict)) + self.l2_coef * (weights @ weights)

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        y_predict = X @ w[1:].T + w[0]
        X = np.hstack( (np.ones( (X.shape[0], 1) ), X) )
        return np.mean(X.T * -y * expit(-y * y_predict), axis = 1) + 2 * self.l2_coef * np.r_[0, w[1:]]