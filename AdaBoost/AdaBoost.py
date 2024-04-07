# Simple AdaBoost Implementation (Initial formulation for a Binary Classification Problemm)

# Core Idea: 
# Sequentially apply a weak learning algorithm to repeatedly 
# modified versions of the data, producing a sequence of weak
# classifiers which are combined into a single classifier that is 
# more accurate than any of the weak classifiers alone

# More Doc: https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/

import numpy as np

# Decision Tree with a Single Split [Weak Learner Class]
class DecisionStump:
    
    def __init__(self) -> None:
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        # Variable for the Performance
        self.alpha = None

    def predict(self, X):
        n_samples, _ = X.shape
        X_column = X[:, self.feature_idx]
        
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            # All the Predictions that are smaller than the threshold have negative 
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = 1

        return predictions


# Class for the AdaBoost Algorithm
class AdaBoost:
    def __init__(self, n_classifiers=5) -> None:
        # Number of Classifiers to Consider [Number of Weak Learners]
        self.n_classifiers = n_classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize the Weights ()
        w = np.full(n_samples, (1/n_samples))

        # List to Store all the Classifiers
        self.classifiers = []

        # Iterate through all the classifiers and update their weights [Training the Model]
        for _ in range(self.n_classifiers):
            # Creating a new weak learner
            clf = DecisionStump()

            # Intializing the Error
            min_error = float('inf')

            for feature_idx in range(n_features):
                X_column = X[:, feature_idx]
                thresholds = np.unique(X_column)

                # Iterate through all the Thresholds
                for threshold in thresholds:
                    # Predict the Polarity and Update the error
                    pol = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    missclassified_weights = w[y != predictions]
                    error = sum(missclassified_weights)

                    # Update Error
                    if error > 0.5:
                        error = 1 - error
                        pol = -1
                    
                    # Found a better classifier
                    if error < min_error:
                        min_error = error
                        clf.polarity = pol
                        clf.threshold = threshold
                        clf.feature_idx = feature_idx

            EPS = 1e-10

            # Calculate the Performance [Calculate Alpha]
            clf.alpha = 0.5 * np.log((1 - error) / (error + EPS))

            # Getting the Predictions
            predictions = clf.predict(X)

            # Updating the Weights
            w *= np.exp(-clf.alpha * y * predictions)
            
            # Normalizing the Weights
            w /= np.sum(w)

            self.classifiers.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.classifiers]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

if __name__ == "__main__":
    
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split

    # def accuracy(y_true, y_pred):
    #     return np.sum(y_true == y_pred) / len(y_true)
    
    # data = datasets.load_breast_cancer()
    # X = data.data
    # y = data.target

    # y[y == 0] = -1

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # clf = AdaBoost(n_classifiers=5)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print("Accuracy = ", accuracy(y_test, y_pred))

    # from openml import study , tasks , runs , extensions, config, exceptions
    # from sklearn import compose , impute , pipeline , preprocessing , tree, metrics
    
    # cont , cat = extensions.sklearn.cont , extensions.sklearn.cat # feature types
    # clf = pipeline.make_pipeline(compose.make_column_transformer(
    #                     (impute. SimpleImputer(), cont),
    #                     (preprocessing.OneHotEncoder(handle_unknown ='ignore'), cat)),
    #                     tree.DecisionTreeClassifier()) # build a classification pipeline
    
    
    # config.apikey = '4ec20fa744d7598e4f6618d15fd370e4'  # set the OpenML Api Key

    # benchmark_suite = study.get_suite('OpenML-CC18') # task collection
    
    # for task_id in benchmark_suite.tasks: # iterate over all tasks
    #     task = tasks.get_task(task_id) # download the OpenML task
    #     run = runs.run_model_on_task(clf , task) # run classifier on splits
    #     score = run.get_metric_score(metrics.accuracy_score)
        
    #     print(score)

    #     # run.publish() # upload the run to the server; optional , requires API key
    #     break

    import openml
    from sklearn import ensemble, neighbors

    openml.config.start_using_configuration_for_example()

    # NOTE: We are using dataset 20 from the test server: https://test.openml.org/d/20
    dataset = openml.datasets.get_dataset(20)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)

    task = openml.tasks.get_task(119)
    clf = ensemble.RandomForestClassifier()
    run = openml.runs.run_model_on_task(clf, task)
    print(run)

    myrun = run.publish()
    print(f"Run was uploaded to {myrun.openml_url}")
    print(f"The flow can be found at {myrun.flow.openml_url}")

    openml.config.stop_using_configuration_for_example()

