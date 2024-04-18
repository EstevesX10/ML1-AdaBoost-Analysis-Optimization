from sklearn.linear_model import (Perceptron)
from .AdaBoost import (AdaBoost)

'''
    # ---------------------- #
    | Modified AdaBoost File |
    # ---------------------- #

This File contains a Modified Version of the AdaBoost Algorithm

'''

class ModifiedAdaBoost(AdaBoost):
    def __init__(self):
        super().__init__()
        self.weak_learner = Perceptron
        self.weak_learner_specs = {}

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba()
    
if __name__ == "__main__":
    d = {'max_depth':1}
