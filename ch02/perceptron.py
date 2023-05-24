###
# page 025
##
import numpy as np


class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.error_ = None
        self.w_ = None
        self.eta = max(0.0, min(eta, 1.0))
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.Shape[1])
        self.error_ = []

        for _ in range(self.n_iter):
            errors = 0
            zip_data = zip(X, y)
            for xi, target in zip_data:
                print(f"{xi}, {target}")
                update = self.eta * (target - self.predict(xi))
                print(f"update {update}")
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.error_.append(errors)
        return self

    def net_input(self, X):
        res = np.dot(X, self.w_[1:]) + self.w_[0]
        return res

    def predict(self, X):
        res = np.where(self.net_input(X) >= 0.0, 1, -1)
        return res
