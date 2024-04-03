import numpy as np

from regularizations import *


class BaseModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def pred(self, X, theta):
        return np.dot(X, theta)

    def train(self, num_epochs, step_size=0.1):
        N, D = self.X.shape
        theta = np.zeros([D, 1])
        loss = 0
        for epoch in range(num_epochs):
            ypred = self.pred(self.X, theta)
            loss = self._mse(self.y, ypred, theta)
            gradient = self._grad(self.X, self.y, theta)
            theta = self._step(theta, gradient, step_size)
        return theta, loss

    def _mse(self, y, ypred, theta):
        raise NotImplementedError

    def _grad(self, X, y, theta):
        raise NotImplementedError

    def _step(self, theta, grad, step_size):
        raise NotImplementedError

    def analytical(self):
        raise NotImplementedError


class LeastSquaresModel(BaseModel, BaseRegularization):
    def _mse(self, y, ypred, _theta=None):
        return np.mean((y - ypred) ** 2)

    def _grad(self, X, y, theta):
        return np.dot(X.T, self.pred(X, theta) - y) / y.shape[0]

    def _step(self, theta, grad, step_size):
        return theta - step_size * grad

    def analytical(self):
        A = np.dot(self.X.T, self.X)
        A = np.linalg.inv(A)
        A = np.dot(A, self.X.T)
        A = np.dot(A, self.y)
        A = A.reshape([A.shape[0], 1])
        return A, self._mse(self.y, self.pred(self.X, A), A)


class RidgeLSModel(BaseModel, RidgeRegularization):
    def __init__(self, X, y, l):
        super().__init__(X, y)
        super(BaseModel, self).__init__(l)
        self.reg = super(BaseModel, self)

    def _mse(self, y, ypred, theta):
        return np.mean((y - ypred) ** 2) + self.cost(theta)

    def _grad(self, X, y, theta):
        return np.dot(X.T, self.pred(X, theta) - y) / y.shape[0] + self.grad(theta)

    def _step(self, theta, grad, step_size):
        return theta - step_size * grad

    def analytical(self):
        A = np.dot(self.X.T, self.X)
        A += self.l * np.identity(A.shape[0], np.int64)
        A = np.linalg.inv(A)
        A = np.dot(A, self.X.T)
        A = np.dot(A, self.y)
        A = A.reshape([A.shape[0], 1])
        return A, self._mse(self.y, self.pred(self.X, A), A)


class LassoLSModel(BaseModel, LassoRegularization):
    def __init__(self, X, y, l):
        super().__init__(X, y)
        super(BaseModel, self).__init__(l)
        self.reg = super(BaseModel, self)

    def _mse(self, y, ypred, theta):
        return np.mean((y - ypred) ** 2) + self.reg.cost(theta)

    def _grad(self, X, y, theta):
        return np.dot(X.T, self.pred(X, theta)-y)/y.shape[0] + self.reg.grad(theta)

    def _step(self, theta, grad, step_size):
        return theta - step_size * grad

    def analytical(self):
        ls_model = LeastSquaresModel(self.X, self.y)
        ls_theta, _ = ls_model.analytical()
        theta =  np.sign(ls_theta)*np.maximum(np.abs(ls_theta)-self.l/2, 0)
        y_pred = self.pred(self.X, theta)
        cost = self._mse(self.y, y_pred, theta)
        return theta, cost

#%%
