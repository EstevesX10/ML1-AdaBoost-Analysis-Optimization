import numpy as np
import pandas as pd
import json
from sklearn.model_selection import (KFold, cross_validate)
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import (roc_curve, roc_auc_score)
from sklearn.metrics import (accuracy_score)
from .DataPreprocessing import (Fetch_X_y)
from .AdaBoost import (AdaBoost)

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
    
    -> def round_numeric(x:float):
        - Formats floats to 2 decimal points (%)

    -> def Evaluate_Models(tasks, models, columns):
        - Calculates the accuracy and standard deviation every model given obtains against all the selected datasets from the 'OpenML-CC18' Study
        and stores the results inside a DataFrame
    
    -> def Evaluate_Average_Results(df:pd.DataFrame, columns:list[str]):
        - Calculates the average accuracy results for each model
    
    -> Calculate_Average_Models_Stats(my_tasks:list[int], models:list[any]):
        - Calculates the average parameters for each model's behaviour

'''

def Save_json_file(content: dict, file_path:str) -> None:
    '''Saves a Dictionary as a json file'''
    with open(file_path, "w") as f:
        json.dump(content , f)
        
def Load_json_file(file_path:str) -> any:
    '''Loads a json file - creates a dictionary'''
    with open(file_path) as f:
        return json.load(f)

def Manage_Results(results:dict) -> np.ndarray:
    '''Converts the Results saved in .json file into a list of points to futher scatterplot'''
    
    # [Results_Diff_1, Results_Diff_2, ...] => as many as the model's processed
    # Results_diff => List of points (x, y) where x = n_weak_learners and y = delta accuracy
    
    try:
        n_models = len(results[3].keys())
    except:
        n_models = len(results['3'].keys())
    
    Processed_Results = [[] for _ in range(n_models)]
    Base_Accuracy = None
    
    for task_id, task_results in results.items():
        for idx, (model, model_results) in enumerate(task_results.items()):
            if (idx == 0):
                Base_Accuracy = model_results['Model_Accuracy']
            else:
                Model_Accuracy = model_results['Model_Accuracy']
                Weak_Learners_Accuracies = model_results['Weak_Learners_Accuracies']
                Random_Guessing_Accuracy = model_results['Majority_Class']
                Delta_Accuracy = (Model_Accuracy - Base_Accuracy)
                N_Weak_Learners_with_less_Accuracy = (sum([Weak_Learner_Accuracy < Random_Guessing_Accuracy for Weak_Learner_Accuracy in Weak_Learners_Accuracies]))
                Processed_Results[idx-1].append((N_Weak_Learners_with_less_Accuracy, Delta_Accuracy))
    
    return Processed_Results[:-1]

def Perform_KFold_CV(X:np.ndarray, y:np.ndarray, model:AdaBoost, total_splits:int=5) -> tuple[np.ndarray, np.ndarray]:
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
    # if (issubclass(AdaBoost, type(model))):
    try:
        assert(model.__name__ in ['AdaBoost [Base]', 'AdaBoost [Max Depth 3]', 'AdaBoost [Perceptron]'])
        for model in AdaBoost_Model_per_Fold:
            weak_learner_accuracy = []
            for weak_learner in model.G_M:
                y_pred = weak_learner.predict(X)
                acc = accuracy_score(y, y_pred)
                weak_learner_accuracy.append(acc)
            AdaBoost_Weak_Learner_Accuracies.append(weak_learner_accuracy)

        # Getting the Mean Accuracy of each weak learner
        AdaBoost_Weak_Learner_Accuracies = np.mean(AdaBoost_Weak_Learner_Accuracies, axis=0)
    except:
        pass

    # Return accuracies
    return AdaBoost_Scores, AdaBoost_Weak_Learner_Accuracies

def Perform_Mean_KFold_CV(X:np.ndarray, y:np.ndarray, model:AdaBoost) -> tuple[float, np.ndarray]:
    # Get accuracies and std's 
    Accuracy, Weak_Learners_Accuracies = Perform_KFold_CV(X, y, model)
    
    # Calculate average accuracy and std
    Avg_Accuracy = np.mean(Accuracy)
    
    return Avg_Accuracy, Weak_Learners_Accuracies

def Evaluate_Model(task_id:int, model:AdaBoost) -> None:
    # Get Features and Target
    ds_name, X, y = Fetch_X_y(task_id)

    # Perform K-Fold Cross Validation
    Avg_Accuracy, Weak_Learners_Accuracies, Best_Model = Perform_Mean_KFold_CV(X, y, model)

    # Print Results
    print(f"[DATASET] {ds_name}\n[Average Accuracy] {Avg_Accuracy:1.3f}")

def round_numeric(x:float) -> float:
    # Function to round numeric values to 2 decimal places
    return round(x*100, 2) if isinstance(x, (int, float)) else x

def Evaluate_Models(tasks:list[int], models:list[AdaBoost], columns:list[str]) -> tuple[pd.DataFrame, dict]:
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

            # Calculating the Average Accuracy
            Average_Model_Accuracy = np.mean(Model_Accuracies)

            # Saving the results to a dictionary [Need to convert the numpy arrays to lists since they are not json serializable]
            Model_Results.update({columns_names[4+idx]:{'Majority_Class':majority_class_cases, 'Model_Accuracy': Average_Model_Accuracy, 'Weak_Learners_Accuracies':list(Weak_Learners_Accuracies)}})

            # Save the Model Mean Accuracy
            dataset_results.append(Average_Model_Accuracy)

        # Updating the results for the current dataset
        Models_Results.update({int(task_id):Model_Results})

        # Update overall Results
        data.append(dataset_results)
        
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns_names)
    df = df.applymap(round_numeric)

    return df, Models_Results

def format_numeric(x:float) -> float:
    # Function to format float values into 2 decimal places
    return round(x, 2) if isinstance(x, (int, float)) else x

def Evaluate_Average_Results(df:pd.DataFrame, columns:list[str]) -> pd.DataFrame:
    # Calculates the average results of each modified model into a new dataframe
    unnecessary_columns = ['Dataset', 'Positive Class (%)', 'Negative Class (%)', 'Majority Class (%)']
    clean_df = df.drop(columns=unnecessary_columns)
    data = []
    for _, col in enumerate(clean_df.columns):
        column_values = clean_df[col].to_numpy()
        data.append(np.mean(column_values))

    # Creating a Final Dataframe
    final_df = pd.DataFrame([data], columns=columns)
    final_df = final_df.applymap(format_numeric)
    return final_df

def Calculate_Average_Models_Stats(my_tasks:list[int], models:list[any]) -> dict:
    # Dictionary to Store the Values
    results = {}

    for my_model in models:
        # Defining arrays to store the information
        false_positive_rates = []
        true_positive_rates = []
        AUCs = []
        model_train_errors = []
        model_alphas = []

        # Defining a linear space to deal with fpr list sizes (due to having multiple datasets with multiple features)
        mean_fpr = np.linspace(0, 1, 100)

        for task_id in my_tasks:
            # Get Features and Target
            ds_name, X, y = Fetch_X_y(task_id)
            (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, stratify=y)
            
            # Creating a new instance of the Model
            model = my_model()
            try:
                model = model.load_model(f"./Experimental Results/Trained Models/{ds_name}_{model.__name__}.jobllib")
            except:
                model.fit(X_train, y_train)
                model.save_model(f"./Experimental Results/Trained Models/{ds_name}_{model.__name__}.jobllib")

            # Calculating the values for the ROC Curve
            y_pred_proba = model.predict_proba(X_test)[:,1]
            model_false_positive_rate, model_true_positive_rate, _ = roc_curve(y_test, y_pred_proba)
            tpr_interpolated = np.interp(mean_fpr, model_false_positive_rate, model_true_positive_rate)
            tpr_interpolated[0] = 0.0
            false_positive_rates.append(mean_fpr)
            true_positive_rates.append(tpr_interpolated)
            model_AUC = roc_auc_score(y_test, y_pred_proba)
            AUCs.append(model_AUC)

            # Getting the weak learners info
            model_train_errors.append(model.training_errors)
            model_alphas.append(model.alphas)
        
        # Calculating the Mean Values
        false_positive_rate = np.mean(false_positive_rates, axis=0)
        true_positive_rate = np.mean(true_positive_rates, axis=0)
        AUC = np.mean(AUCs)
        train_errors = np.mean(model_train_errors, axis=0)
        alphas = np.mean(model_alphas, axis=0)
        
        # Updating Results
        results.update({model.__name__:{'fpr':list(false_positive_rate),
                                        'tpr':list(true_positive_rate),
                                        'AUC':float(AUC),
                                        'train_errors':list(train_errors),
                                        'alphas':list(alphas)}})

    return results