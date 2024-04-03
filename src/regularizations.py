import numpy as np


class BaseRegularization:
    def cost(self, theta):
        raise NotImplementedError

    def grad(self, theta):
        raise NotImplementedError


class RidgeRegularization(BaseRegularization):
    def __init__(self, l):
        self.l = l

    def cost(self, theta):
        return self.l * np.sum(theta ** 2)

    def grad(self, theta):
        return 2 * theta


class LassoRegularization(BaseRegularization):
    def __init__(self, l):
        self.l = l

    def cost(self, theta):
        return self.l * np.sum(np.abs(theta))

    def grad(self, theta):
        return np.sign(theta)
