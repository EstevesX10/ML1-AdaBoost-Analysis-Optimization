import openml
import numpy as np
import pandas as pd

'''
    # ----------------------- #
    | Data Preprocessing File |
    # ----------------------- #

This File contains multiple functions used to Process Data:

    -> Fetch_Dataset(task_id):
        - Gets the dataset in dataframe format from a given task
    
    -> Fetch_X_y(task_id):
        - Gets the arrays of Features and Targets from a dataset belonging to a given task
'''

def Fetch_Dataset(task_id):
    '''Fetches a Dataset given a OpenML Task ID'''
    # Get OpenML task
    task = openml.tasks.get_task(task_id)

    # Get Dataset
    dataset = openml.datasets.get_dataset(task.dataset_id)

    # Getting the Data
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                               target=dataset.default_target_attribute)
    # Creating a Dataframe with the Data
    df = pd.DataFrame(X, columns=attribute_names)
    
    # Convert Target Values ({0, 1}) into {-1, 1}:
    df['target'] = (2*y - 1).astype(np.int8)
    
    # Remove rows with NaN values
    df = df.dropna(how='any', axis=0)
    
    return dataset.name, df

def Fetch_X_y(task_id):
    '''Fetches a Dataset given a OpenML Task ID'''
    # Get OpenML task
    task = openml.tasks.get_task(task_id)

    # Get Adjacent Dataset
    dataset = openml.datasets.get_dataset(task.dataset_id)

    # Getting the Data
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array",
                                                                    target=dataset.default_target_attribute)
    
    # Creating a Dataframe with the Data
    df = pd.DataFrame(X, columns=attribute_names)
    df['target'] = y
    
    # Remove rows with NaN values
    df = df.dropna(how='any', axis=0)

    # Split the Dataframe once again
    cols = list(df.columns)
    cols.remove('target')
    X = np.array(df[cols])
    y = np.array(df['target'])
    
    # Convert Target Values ({0, 1}) into {-1, 1}:
    y = (2*y - 1).astype(np.int8)
    
    return dataset.name, X, y