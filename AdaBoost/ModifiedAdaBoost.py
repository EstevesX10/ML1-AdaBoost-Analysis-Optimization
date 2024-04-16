import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from AdaBoost import (AdaBoost)

class ModifiedAdaboost(AdaBoost):
    
    def __init__(self):
        super.__init__()
        self.weak_learner = DecisionTreeClassifier
        self.weak_learner_specs = {'max_depth':1}

    def fit(self, X, y, M = 100):
        return super().fit(X, y, M)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba()