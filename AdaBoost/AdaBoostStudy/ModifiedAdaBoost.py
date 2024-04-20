from sklearn.linear_model import (Perceptron)
from sklearn.tree import (DecisionTreeClassifier)
from .AdaBoost import (AdaBoost)

'''
    # ---------------------- #
    | Modified AdaBoost File |
    # ---------------------- #

This File contains a Modified Version of the AdaBoost Algorithm

'''

class AdaBoostPerceptron(AdaBoost):
    def __init__(self):
        super().__init__()
        self.weak_learner = Perceptron
        self.weak_learner_specs = {}

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoostTunedDT(AdaBoost):
    def __init__(self, hyperparameters):
        super().__init__()
        self.weak_learner = DecisionTreeClassifier
        self.weak_learner_specs = hyperparameters

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)