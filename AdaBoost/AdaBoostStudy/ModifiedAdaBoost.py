from sklearn.tree import (DecisionTreeClassifier)
from sklearn.linear_model import (Perceptron)
from .AdaBoost import (AdaBoost)
from sklearn.neural_network import (MLPClassifier)

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

class AdaBoostPerceptron(AdaBoost):
    def __init__(self):
        super().__init__(weak_learner=Perceptron, weak_learner_hyperparameters={})

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

class AdaBoostMLP(AdaBoost):
    def __init__(self):
        super().__init__(weak_learner=MLPClassifier, weak_learner_hyperparameters={'hidden_layer_sizes':(1,), 'early_stopping':True})

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)
