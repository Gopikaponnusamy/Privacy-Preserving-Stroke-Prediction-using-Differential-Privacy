import numpy as np
from sklearn.linear_model import LogisticRegression

class DPLogisticRegression:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        self.model = LogisticRegression(max_iter=1000)

    def add_noise(self, weights):
        noise = np.random.normal(0, 1/self.epsilon, weights.shape)
        return weights + noise

    def fit(self, X, y):
        self.model.fit(X, y)
        self.model.coef_ = self.add_noise(self.model.coef_)

    def predict(self, X):
        return self.model.predict(X)