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

def compute_logistic_loss(y_true, y_pred):
    # Assuming y_true is in {-1, 1}
    return np.log(1 + np.exp(-2 * y_true * y_pred))

def compute_logistic_loss_derivative(y_true, y_pred):
    return -2 * y_true / (1 + np.exp(2 * y_true * y_pred))

def compute_alpha(y_true, y_pred, w_i):
    """
    Compute the alpha (weight) of a weak learner given its error rate, while avoiding
    numerical issues that can lead to NaN values.
    
    Parameters:
    - y_true: numpy array, true labels
    - y_pred: numpy array, predicted labels by the weak learner
    - w_i: numpy array, weights for each instance in the dataset

    Returns:
    - alpha: the weight of the weak learner
    """
    # Calculate error rate with a safeguard against zero division or extreme values
    error = np.dot(w_i, y_true != y_pred) / np.sum(w_i)
    error = np.clip(error, 1e-10, 1 - 1e-10)  # Ensure error is never 0 or 1

    # Calculate alpha using a stable logarithmic transformation
    try:
        alpha = np.log((1 - error) / error)
    except OverflowError:
        # Handle potential overflow issues by capping alpha to a large positive number
        alpha = 100.0 if error < 0.5 else -100.0

    # Check for NaN after calculation (unlikely after safeguards, but safe to check)
    if np.isnan(alpha):
        raise ValueError(f"Alpha calculation resulted in NaN for error rate: {error}")

    return alpha

def update_weights(w_i, alpha, y_true, y_pred):
    exponent = -alpha * compute_logistic_loss_derivative(y_true, y_pred)
    exponent = np.clip(exponent, a_min=-100, a_max=100)  # Avoid extremely large values
    w_i *= np.exp(exponent)
    w_i /= np.sum(w_i)
    return w_i

class AdaBoost_LogisticLoss(AdaBoost):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, M = 100):
        '''
        Fit model. Arguments:
        X: independent variables - array-like matrix
        y: target variable - array-like vector
        M: number of boosting rounds. Default is 100 - integer
        '''
        
        # Clear before calling
        self.alphas = []
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(M):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = self.weak_learner(**self.weak_learner_hyperparameters)  # By Default uses a Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error - CHANGED
            error_m = compute_logistic_loss(y, y_pred)
            self.training_errors.append(error_m)

            # (c) Compute alpha - CHANGED
            alpha_m = compute_alpha(error_m, y, y_pred)
            self.alphas.append(alpha_m)

            print(self.alphas)

        assert len(self.G_M) == len(self.alphas)
        self.classes_ = np.unique(y_pred)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)