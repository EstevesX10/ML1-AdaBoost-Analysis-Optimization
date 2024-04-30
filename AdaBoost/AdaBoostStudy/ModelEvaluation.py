import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, cross_val_score)
from .DataPreprocessing import (Fetch_X_y)

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
    
    -> Evaluate_Model_AllDS(tasks, model):
        - Calculates the accuracy and standard deviation a given model obtains against all the selected datasets from the 'OpenML-CC18' Study 
    
    -> def Evaluate_Models(tasks, models, columns):
        - Calculates the accuracy and standard deviation every model given obtains against all the selected datasets from the 'OpenML-CC18' Study
        and stores the results inside a DataFrame 

'''

def Perform_KFold_CV(X, y, model, total_splits=5):
    # Set the K-Fold cross-validation
    kf = KFold(n_splits=total_splits, shuffle=True, random_state=123)
    
    # Perform K-Fold cross-validation
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)

    # Return accuracies and respective standard deviations
    return scores, scores.std()

def Perform_Mean_KFold_CV(X, y, model):
    # Get accuracies and std's 
    Accuracy, Std = Perform_KFold_CV(X, y, model)
    
    # Calculate average accuracy and std
    Avg_Accuracy = np.mean(Accuracy)
    Avg_Std = np.mean(Std)
    
    return Avg_Accuracy, Avg_Std

def Evaluate_Model(task_id, model):
    # Get Features and Target
    ds_name, X, y = Fetch_X_y(task_id)

    # Perform K-Fold Cross Validation
    Avg_Accuracy, Avg_Std = Perform_Mean_KFold_CV(X, y, model)

    # Print Results
    print(f"[DATASET] {ds_name}\n[Average Accuracy] {Avg_Accuracy:1.3f} +/- {Avg_Std:1.3f}")

def Evaluate_Model_AllDS(tasks, model):
    # Evaluate all Datasets Retrieved
    for task_id in tasks.keys():
        Evaluate_Model(task_id, model)
        print()

def Evaluate_Models(tasks, models, columns):
    # Create List to store the obtained results 
    data = []

    columns_names = ['Dataset', 'Positive Cases (%)', 'Negative Cases (%)', 'Majority Class (%)'] + columns

    for task_id in tasks.keys():
        # Get Features and Target
        ds_name, X, y = Fetch_X_y(task_id)

        positive_cases = sum(y == 1) / len(y)
        negative_cases = sum(y == -1) / len(y)
        majority_class_cases = max(positive_cases, negative_cases)

        # Initialize results list for the current dataset
        dataset_results = [ds_name, positive_cases, negative_cases, majority_class_cases]

        for model in models:
            # Perform K-Fold Cross Validation
            Avg_Accuracy, Avg_Std = Perform_Mean_KFold_CV(X, y, model)

            # Save Results
            dataset_results.append(Avg_Accuracy)

        # Update overall Results
        data.append(dataset_results)
        
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns_names)

    return df