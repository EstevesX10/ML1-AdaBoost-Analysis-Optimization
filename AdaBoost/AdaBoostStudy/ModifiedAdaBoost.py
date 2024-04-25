import numpy as np
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
        super().__init__(weak_learner=Perceptron, weak_learner_hyperparameters={})

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoostTunedDT(AdaBoost):
    def __init__(self, weak_learner_hyperparameters={'max_depth':1}):
        super().__init__(weak_learner=DecisionTreeClassifier, weak_learner_hyperparameters=weak_learner_hyperparameters)

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

'''
-------------------------------------------------------
'''

class AdaBoost_LogisticLoss(AdaBoost):
    def __init__(self):
        super().__init__(loss_function='logistic')

    def compute_error(self, y_true, y_pred, w_i):
        return self.compute_error(y_true, y_pred, w_i)

    def compute_alpha(self, error):
        return super().compute_alpha(error)

    def update_weights(self, w_i, alpha, y_true, y_pred):
        return super().update_weights(w_i, alpha, y_true, y_pred)

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)