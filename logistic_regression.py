import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self,
            x_train: np.ndarray, y_train: np.ndarray,
            max_iter: int = 100):
        w = np.zeros(np.size(x_train, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self.sigmoid(x_train @ w)
            grad = x_train.T @ (y - y_train)
            hessian = (x_train.T * y * (1 - y)) @ x_train
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def proba(self, x: np.ndarray):
        return self.sigmoid(x @ self.w)

    def classify(self, x: np.ndarray, threshold: float = 0.5):
        return (self.proba(x) > threshold).astype(int)
