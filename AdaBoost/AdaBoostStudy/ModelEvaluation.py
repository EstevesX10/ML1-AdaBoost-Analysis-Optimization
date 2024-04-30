import numpy as np
import pandas as pd
import json
from sklearn.model_selection import (KFold, cross_val_score, cross_validate)
from .DataPreprocessing import (Fetch_X_y)
from sklearn.metrics import (accuracy_score)
from .AdaBoost import AdaBoost

'''
    # --------------------- #
    | Model Evaluation File |
    # --------------------- #

This File contains multiple functions used to Evaluate the Machine Learning Models developed:

    -> Perform_KFold_CV(X, y, model):
        - Perfoms K-Fold Cross Validation returning 2 arrays for the accuracies and standard deviations
    
    -> Perform_Mean_KFold_CV(X, y, model):
        - Performs a K-Fold Cross Validation but returns the mean values of the accuracy and standard deviation
    
    -> Evaluate_Model(task_id, model):
        - Calculates the accuracy and standard deviation a given model obtains against a given task
    
    -> def Evaluate_Models(tasks, models, columns):
        - Calculates the accuracy and standard deviation every model given obtains against all the selected datasets from the 'OpenML-CC18' Study
        and stores the results inside a DataFrame 

'''

def Save_json_file(content: dict, file_path:str):
    '''Saves a Dictionary as a json file'''
    with open(file_path, "w") as f:
        json.dump(content , f)
        
def Load_json_file(file_path:str):
    '''Loads a json file - creates a dictionary'''
    with open(file_path) as f:
        return json.load(f)

def Perform_KFold_CV(X, y, model:AdaBoost, total_splits=5):
    # Set the K-Fold cross-validation
    kf = KFold(n_splits=total_splits, shuffle=True, random_state=123)
    
    # Perform K-Fold cross-validation
    AdaBoost_Study = cross_validate(model, X, y, cv=kf, return_estimator=True)

    # Getting the Accuracies of the models per Fold
    AdaBoost_Scores = AdaBoost_Study['test_score']

    # Getting the Models per Fold
    AdaBoost_Model_per_Fold = AdaBoost_Study['estimator']

    # List to hold the calculated accuracies of the Weak Learners
    AdaBoost_Weak_Learner_Accuracies = []
    
    # Iterating through all the models and calculate their weak learner's accuracies
    if (issubclass(AdaBoost, type(model))):
        for model in AdaBoost_Model_per_Fold:
            weak_learner_accuracy = []
            for weak_learner in model.G_M:
                y_pred = weak_learner.predict(X)
                acc = accuracy_score(y, y_pred)
                weak_learner_accuracy.append(acc)
            AdaBoost_Weak_Learner_Accuracies.append(weak_learner_accuracy)

        # Getting the Mean Accuracy of each weak learner
        AdaBoost_Weak_Learner_Accuracies = np.mean(AdaBoost_Weak_Learner_Accuracies, axis=0)

    # Return accuracies and respective standard deviations
    return AdaBoost_Scores, AdaBoost_Weak_Learner_Accuracies

def Perform_Mean_KFold_CV(X, y, model):
    # Get accuracies and std's 
    Accuracy, Weak_Learners_Accuracies = Perform_KFold_CV(X, y, model)
    
    # Calculate average accuracy and std
    Avg_Accuracy = np.mean(Accuracy)
    
    return Avg_Accuracy, Weak_Learners_Accuracies

def Evaluate_Model(task_id, model):
    # Get Features and Target
    ds_name, X, y = Fetch_X_y(task_id)

    # Perform K-Fold Cross Validation
    Avg_Accuracy, Weak_Learners_Accuracies = Perform_Mean_KFold_CV(X, y, model)

    # Print Results
    print(f"[DATASET] {ds_name}\n[Average Accuracy] {Avg_Accuracy:1.3f}")

def Evaluate_Models(tasks, models, columns):
    # Create List to store the obtained results 
    data = []
    Models_Results = {}
    columns_names = ['Dataset', 'Positive Class (%)', 'Negative Class (%)', 'Majority Class (%)'] + columns

    for task_id in tasks.keys():
        # Get Features and Target
        ds_name, X, y = Fetch_X_y(task_id)

        # Calculating the classes cardinality inside the dataset
        positive_cases = sum(y == 1) / len(y)
        negative_cases = sum(y == -1) / len(y)
        majority_class_cases = max(positive_cases, negative_cases)

        # Initialize results list for the current dataset
        dataset_results = [ds_name, positive_cases, negative_cases, majority_class_cases]

        # Dictionary to store the calculated accuracies as well as the accuracies for each weak learner within the main model
        Model_Results = {}

        for idx, model in enumerate(models):
            # Perform K-Fold Cross Validation
            Model_Accuracies, Weak_Learners_Accuracies = Perform_KFold_CV(X, y, model)

            # Saving the results to a dictionary [Need to convert the numpy arrays to lists since they are not json serializable]
            Model_Results.update({columns_names[4+idx]:{'Model_Accuracies': list(Model_Accuracies), 'Weak_Learners_Accuracies':list(Weak_Learners_Accuracies)}})

            # Save the Model Mean Accuracy
            dataset_results.append(np.mean(Model_Accuracies))

        # Updating the results for the current dataset
        Models_Results.update({int(task_id):Model_Results})

        # Update overall Results
        data.append(dataset_results)
        
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns_names)

    return df, Models_Results