import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = None

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        
        def learning_rate(k, alpha, beta):
            """
            function for calculating learning_rate on each iteration 
            k: int 
            alpha, beta: float
            """
            return alpha / (k ** beta)

        np.random.seed(self.random_seed)
        
        if X_val is None:
            X_val = X.copy()

        if y_val is None:
            y_val = y.copy()

        if w_0 is None:
            w_0 = np.random.normal(size = X.shape[1])
            w_0 = np.hstack( (1, w_0) )

        if trace:
            history = {'time': [],'func': [], 'func_val': []}

        w = w_0.copy()
        loss_func = self.loss_function.func(X, y, w)

        for i in range (1, self.max_iter + 1):
            begin = time.time()
            lr = learning_rate(i, self.step_alpha, self.step_beta) # learning rate

            if self.batch_size >= X.shape[0]:
                w -= lr * self.loss_function.grad(X, y, w)

            else:
                str_numbers = np.arange(X.shape[0])
                np.random.shuffle(str_numbers)
                tmp = 0
                for j in range(X.shape[0] // self.batch_size):
                    mask = str_numbers[tmp : min(tmp + self.batch_size, X.shape[0])]
                    X_train_mini = X[mask]
                    y_train_mini = y[mask]
                    w -= lr * self.loss_function.grad(X_train_mini, y_train_mini, w)
                    tmp += self.batch_size
            new_loss_func = self.loss_function.func(X, y, w)
            if abs(new_loss_func - loss_func) < self.tolerance:
                break

            loss_func = new_loss_func
            end = time.time()

            if trace:
                history['time'].append(end-begin)
                history['func'].append(loss_func)
                history['func_val'].append(self.loss_function.func(X_val, y_val, w))

        self.w = w
        return history if trace else None


    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        X = np.hstack( (np.ones( (X.shape[0], 1) ), X) )
        probability = expit(X @ self.w.T)
        prediction = np.ones(X.shape[0])
        prediction[probability > threshold] = -1
        return prediction
        

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.w)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]
