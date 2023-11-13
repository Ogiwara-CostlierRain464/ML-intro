import numpy as np


class LinearRegression:
    def __init__(self):
        self.var = None
        self.w = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        self.w = np.linalg.pinv(x_train) @ y_train
        self.var = np.mean(np.square(x_train @ self.w - y_train))

    def predict(self, x: np.ndarray):
        y = x @ self.w
        return y
