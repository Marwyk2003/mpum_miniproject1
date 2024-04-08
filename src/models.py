import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class BaseModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def pred(self, X, theta):
        return np.dot(X, theta)

    def train(self, num_epochs, step_size=0.1, batch_size=100):
        N, D = self.X.shape
        theta = np.zeros([D, 1])
        loss = 0
        thetas = []
        for epoch in range(num_epochs):
            gradient = self._grad(self.X, self.y, theta)
            theta = self._step(theta, gradient, step_size)
            ypred = self.pred(self.X, theta)
            loss = self.mse(self.y, ypred, theta)
            thetas += [theta]

        return thetas, loss

    def mse(self, y, ypred, theta):
        raise NotImplementedError

    def _grad(self, X, y, theta):
        raise NotImplementedError

    def _step(self, theta, grad, step_size):
        return theta - step_size * grad

    def analytical(self):
        raise NotImplementedError


class LeastSquaresModel(BaseModel):
    def mse(self, y, ypred, _theta=None):
        return np.sum((y - ypred) ** 2) / y.shape[0]

    def _grad(self, X, y, theta):
        return np.dot(X.T, self.pred(X, theta) - y) / y.shape[0]

    def analytical(self):
        A = np.dot(self.X.T, self.X)
        A = np.linalg.inv(A)
        A = np.dot(A, self.X.T)
        A = np.dot(A, self.y)
        A = A.reshape([A.shape[0], 1])

        ypred = self.pred(self.X, A)
        loss = self.mse(self.y, ypred)
        return A, loss


class RidgeLSModel(BaseModel):
    def __init__(self, X, y, l):
        super().__init__(X, y)
        self.l = l

    def mse(self, y, ypred, theta):
        return np.sum((y - ypred) ** 2) / y.shape[0] + self.l * np.sum(theta[1:] ** 2)

    def _grad(self, X, y, theta):
        reg_grad = self.l * 2 * theta
        reg_grad[0] = 0
        return (np.dot(X.T, self.pred(X, theta) - y)) / y.shape[0] + reg_grad

    def analytical(self):
        A = np.dot(self.X.T, self.X)
        A += np.identity(self.X.shape[1]) * self.l
        A = np.linalg.inv(A)
        A = np.dot(A, self.X.T)
        A = np.dot(A, self.y)
        A = A.reshape([A.shape[0], 1])

        ypred = self.pred(self.X, A)
        loss = self.mse(self.y, ypred, A)
        return A, loss


class LassoLSModel(BaseModel):
    def __init__(self, X, y, l):
        super().__init__(X, y)
        self.l = l

    def mse(self, y, ypred, theta):
        return np.sum((y - ypred) ** 2) / y.shape[0] + self.l * np.sum(np.abs(theta[1:]))

    def _convert(self, ls_theta):
        return np.sign(ls_theta) * np.maximum(np.abs(ls_theta) - self.l / 2, 0)

    def _grad(self, X, y, theta):
        reg_grad = self.l * np.sign(theta)
        reg_grad[0] = 0
        return np.dot(X.T, self.pred(X, theta) - y) / y.shape[0] + reg_grad


class ElasticLSModel(BaseModel):
    def __init__(self, X, y, l1, l2):
        super().__init__(X, y)
        self.l1 = l1
        self.l2 = l2

    def mse(self, y, ypred, theta):
        return np.mean((y - ypred) ** 2) + self.l1 * np.sum(np.abs(theta[1:])) + self.l2 * np.sum(theta[1:] ** 2)

    def _convert(self, ls_theta):
        return np.sign(ls_theta) * np.maximum(np.abs(ls_theta) - self.l1 / 2, 0)

    def _grad(self, X, y, theta):
        reg_grad = self.l1 * np.sign(theta) + self.l2 * 2 * theta
        reg_grad[0] = 0
        return np.dot(X.T, self.pred(X, theta) - y) / y.shape[0]
