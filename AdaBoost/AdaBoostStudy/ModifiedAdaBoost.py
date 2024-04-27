import numpy as np
from sklearn.tree import (DecisionTreeClassifier)
from sklearn.svm import (SVC)
from sklearn.linear_model import (Perceptron)
from .AdaBoost import (AdaBoost)

'''
    # ---------------------- #
    | Modified AdaBoost File |
    # ---------------------- #

This File contains Modified Versions of the AdaBoost Algorithm based on:

    -> Weak Learner Choice
    -> Loss Functions used in training

'''

class AdaBoostTunedDT(AdaBoost):
    def __init__(self, weak_learner_hyperparameters={'max_depth':1}):
        super().__init__(weak_learner=DecisionTreeClassifier, weak_learner_hyperparameters=weak_learner_hyperparameters)

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoostSVM(AdaBoost):
    def __init__(self):
        super().__init__(weak_learner=SVC, weak_learner_hyperparameters={'kernel':'linear'})

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoostPerceptron(AdaBoost):
    def __init__(self):
        super().__init__(weak_learner=Perceptron, weak_learner_hyperparameters={})

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

# -------------------------------------------------------

class AdaBoost_LogisticLoss(AdaBoost):
    def __init__(self):
        super().__init__(loss_function = 'logistic')

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoost_HingeLoss(AdaBoost):
    def __init__(self):
        super().__init__(loss_function = 'hinge')

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoost_SquaredLoss(AdaBoost):
    def __init__(self):
        super().__init__(loss_function = 'squared')

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)