import numpy as np
from sklearn.tree import (DecisionTreeClassifier)
from sklearn.linear_model import (Perceptron)
from .AdaBoost import (AdaBoost)


'''
    # ---------------------- #
    | Modified AdaBoost File |
    # ---------------------- #

This File contains Modified Versions of the AdaBoost Algorithm based on:

    -> Weak Learner Choice [AdaBoostTunedDT, AdaBoostPerceptron]

'''

class AdaBoostTunedDT(AdaBoost):
    def __init__(self, weak_learner_hyperparameters={'max_depth':1}) -> None:
        super().__init__(weak_learner=DecisionTreeClassifier, weak_learner_hyperparameters=weak_learner_hyperparameters)

    def fit(self, X:np.ndarray, y:np.ndarray, M = 100) -> None:
        return super().fit(X, y, M)

    def predict(self, X:np.ndarray) -> np.ndarray:
        return super().predict(X)

    def predict_proba(self, X:np.ndarray) -> np.ndarray:
        return super().predict_proba(X)
    
    def save_model(self, file_path:str) -> None:
        return super().save_model(file_path)
    
    def load_model(self, file_path:str) -> any:
        return super().load_model(file_path)

class AdaBoostPerceptron(AdaBoost):
    def __init__(self) -> None:
        super().__init__(weak_learner=Perceptron, weak_learner_hyperparameters={})

    def fit(self, X:np.ndarray, y:np.ndarray, M = 100) -> None:
        return super().fit(X, y, M)

    def predict(self, X:np.ndarray) -> np.ndarray:
        return super().predict(X)

    def predict_proba(self, X:np.ndarray) -> np.ndarray:
        return super().predict_proba(X)
    
    def save_model(self, file_path:str) -> None:
        return super().save_model(file_path)
    
    def load_model(self, file_path:str) -> any:
        return super().load_model(file_path)