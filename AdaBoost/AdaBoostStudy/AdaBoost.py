import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, ClassifierMixin)
from sklearn.tree import (DecisionTreeClassifier)

'''
    # ------------------ #
    | Base AdaBoost File |
    # ------------------ #

This File contains the Base AdaBoost Algorithm used for Binary Classification Problems
as well as some functions to compute values along the training and testing of the model 

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

__version__ = "2.0.1"

# Define AdaBoost class
class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, weak_learner=None, weak_learner_hyperparameters=None, loss_function=None, learning_rate=None):
        self.alphas = []
        self.weak_learner = DecisionTreeClassifier if weak_learner is None else weak_learner
        self.weak_learner_hyperparameters = {'max_depth':1} if weak_learner_hyperparameters is None else weak_learner_hyperparameters
        self.loss_function = 'exponential' if loss_function is None else loss_function
        self.learning_rate = 0.01 if learning_rate is None else learning_rate
        
        self.G_M = []
        self.M = None

        self.training_errors = []
        self.prediction_errors = []

    def compute_error(self, y_true, y_pred, w_i):
        '''
        Calculate the error rate of a weak classifier m. Arguments:
        y: actual target value
        y_pred: predicted value by weak classifier
        w_i: individual weights for each observation
        
        Note that all arrays should be the same length
        '''
        # Assuming y_true is in {-1, 1}
        return np.sum(w_i * (y_pred != y_true))

    def compute_alpha(self, error):
        '''
        Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
        alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
        error: error rate from weak classifier m
        '''
        
        # Exponential Loss and Logistic Loss
        if self.loss_function == 'exponential' or self.loss_function == 'logistic':
            return 0.5 * np.log((1 - error) / error)
        
        # Squared Loss and Hinge Loss
        elif self.loss_function == 'hinge' or self.loss_function == 'squared':
            return self.learning_rate * (1 - error)

        else:
            raise ValueError("Unsupported Loss Function!")

    def update_weights(self, w_i, alpha, y_true, y_pred):
        ''' 
        Update individual weights w_i after a boosting iteration. Arguments:
        w_i: individual weights for each observation
        y: actual target value
        y_pred: predicted value by weak classifier  
        alpha: weight of weak classifier used to estimate y_pred
        '''
        
        # Default Exponential Loss
        if self.loss_function == 'exponential':  
            w_i *= np.exp(alpha * (np.not_equal(y_true, y_pred)).astype(int))
        
        # Logistic Loss
        elif self.loss_function == 'logistic':
            w_i *= np.log(1 + np.exp(alpha * (np.not_equal(y_true, y_pred)).astype(int)))
        
        # Hinge Loss
        elif self.loss_function == 'hinge':
            incorrect = (y_true * y_pred < 1)
            w_i[incorrect] *= np.exp(alpha)

        # Squared Loss
        elif self.loss_function == 'squared':
            w_i *= np.exp(alpha * (1 - y_true * y_pred)**2)
        
        else:
            raise ValueError("Unsupported Loss Function!")
        
        # Normalize weights
        return w_i / np.sum(w_i)

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
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = self.weak_learner(**self.weak_learner_hyperparameters)  # By Default uses a Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)
        self.classes_ = np.unique(y_pred)

    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables - array-like
        '''

        assert(len(self.G_M) > 0)

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Calculate final predictions
        y_pred = (np.sign(weak_preds.T.sum())).astype(int)

        return y_pred

    def predict_proba(self, X):
        '''
        Predict class probabilities using the fitted model.
        X: independent variables - array-like matrix
        '''

        # Initialize dataframe with weak predictions for each observation, weighted by alpha_m
        weighted_sums = np.zeros(len(X))

        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X)
            weighted_sums += y_pred_m * self.alphas[m]

        # Calculate probabilities using the sigmoid function
        proba = sigmoid(weighted_sums)
        
        # Since the class labels are in {-1, 1}
        return np.vstack([(1 - proba), proba]).T