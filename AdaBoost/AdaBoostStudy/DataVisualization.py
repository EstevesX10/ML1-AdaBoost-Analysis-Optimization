import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score)
import scikit_posthocs as sp
import statsmodels.api as sm
from .AdaBoost import (AdaBoost)
from .ModelEvaluation import (Perform_KFold_CV)

'''
    # ----------------------- #
    | Data Visualization File |
    # ----------------------- #

This File contains multiple functions used to Visualize Data:

    -> Compare_Models_Accuracies(X, y, Models, ModelsNames, Colors):
        - Plots the Accuracies of given Models throughout K-Fold Cross Validation

    -> Display_Confusion_Matrix(fit_model, X_Test, y_Test, labels):
        - Displays a Confusion Matrix of a given Model
    
    -> Plot_ROC_Curve(fit_model, X_Test, Y_Test):
        - Plots the ROC Curve

    -> Plot_Weak_Learners_Stats(FitModel):
        - Plots Statistics regarding the Variance of the Weak Learner's Error during Training as well as their weights throughout training

    -> Plot_Model_Stats(FitModel, X_Test, Y_Test):
        - Plots The Model's weak learner's Training Error and Weights over N Boosting Rounds as well as the ROC Curve and the Confusion Matrix

    -> Compare_Models_Stats(FitModels, ModelsNames, X_test, y_test):
        - Plots Statiscal Data (Error in Training, Weights and ROC Curve) for a list of given Trained Models - Allow for parallel comparison

    -> Model_Accuracies_Per_Dataset(results, model_name):
        - Plots a Barplot with the Model's Accuracies per Dataset previously calculated - Inside Results
    
    -> Plot_Critial_Difference_Diagram(Matrix, Colors):
        - Plots the Critical Difference Diagram
'''

def Plot_Scatterplot(Points:list[tuple], Title:str='Insert Title', y_label:str='Insert y Label', x_label:str='Insert X Label') -> None:
   # Unpacking the points into x and y coordinates
    x_values, y_values = zip(*Points)
    
    # Create the scatter plot
    plt.scatter(x_values, y_values, color='#1F7799')
    
    # Using LOWESS to fit a smooth line through the points
    lowess = sm.nonparametric.lowess(y_values, x_values, frac=0.35)  # Adjust frac to change the smoothness
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # Plotting the LOWESS result
    plt.plot(lowess_x, lowess_y, '#AF1021', linestyle='--')
    
    # Adding titles and labels
    plt.title(Title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Displaying the plot
    plt.show()

def Plot_Scatterplots(Data:list[list[tuple]], Titles:list[str]=None, y_label:str=None, x_label:str=None) -> None:
    # Getting total of graphs
    n_plots = len(Data)

    # Setting number of rows and cols
    if (n_plots > 3):
        n_cols = 3
        n_rows = (n_plots // 3) + 1
    else:
        n_cols = n_plots % 3
        n_rows = 1

    # Setting Default Values
    Titles = ['Insert Title' for _ in range(n_plots)] if Titles is None else Titles
    y_label = 'Insert y Label' if y_label is None else y_label
    x_label = 'Insert X Label' if x_label is None else x_label

    # Create a figure
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*12, n_cols*3))

    axs_used = []

    if (n_rows == 1):
        # Create a figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*10, n_cols*2))

        for idx, Points in enumerate(Data):
            # Unpacking values
            x, y = zip(*Points)

            # Scatter the points
            axs[idx].scatter(x, y, color='#1F7799')

            # Using LOWESS to fit a smooth line through the points [Adjust frac to change the smoothness]
            lowess = sm.nonparametric.lowess(y, x, frac=0.35)  
            lowess_x = list(zip(*lowess))[0]
            lowess_y = list(zip(*lowess))[1]

            # Add a few more details to the Plot
            axs[idx].plot(lowess_x, lowess_y, '#AF1021', linestyle='--')
            axs[idx].set_title(Titles[idx])
            axs[idx].set_xlabel(x_label)
            axs[idx].set_ylabel(y_label)

            axs_used.append((idx, 0))

        for i in range(n_rows):
            if ((i, 0) not in axs_used):
                axs[i, 0].axis('off')

    else:
        # Create a figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows*4, n_cols*n_rows))

        for idx, Points in enumerate(Data):
            # Unpacking values
            x, y = zip(*Points)

            # Scatter the points
            axs[idx//3, idx%3].scatter(x, y, color='#1F7799')

            # Using LOWESS to fit a smooth line through the points
            lowess = sm.nonparametric.lowess(y, x, frac=0.35)  # Adjust frac to change the smoothness
            lowess_x = list(zip(*lowess))[0]
            lowess_y = list(zip(*lowess))[1]

            # Add a few more details to the Plot
            axs[idx//3, idx%3].plot(lowess_x, lowess_y, '#AF1021', linestyle='--')
            axs[idx//3, idx%3].set_title(Titles[idx])
            axs[idx//3, idx%3].set_xlabel(x_label)
            axs[idx//3, idx%3].set_ylabel(y_label)

            axs_used.append((idx//3, idx%3))

        for i in range(n_rows):
            for j in range(n_cols):
                if ((i, j) not in axs_used):
                    axs[i, j].axis('off')

    # Adjust Layout
    plt.tight_layout()

    # Displaying the plot
    plt.show()


# ------------------------------------------------------

def Display_Confusion_Matrix(FitModel:AdaBoost, X_Test:np.ndarray, y_Test:np.ndarray, labels:np.ndarray) -> None:

    '''
    Displays the Model's Confusion Matrix.
    FitModel := Trained AdaBoost Classifier 
    X_Test := Array with the Feature's Test set  
    Y_Test := Array with the Label's Test set
    labels := labels used in the X and Y axis description
    '''

    # Creating a Confusion Matrix
    cm = confusion_matrix(y_Test, FitModel.predict(X_Test))
    
    # Adds an Axes to the current figure 
    # [Check Matplotlib Documentation at: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html]
    ax = plt.subplot()

    # Creating a HeatMap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)

    # Setting the Title
    ax.set_title('Confusion Matrix')

    # Setting the names to the X & Y labels
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
    # Inserting the class labels into the X & Y Axis
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Showing Final Product
    plt.show()

def Plot_ROC_Curve(FitModel:AdaBoost, X_Test:np.ndarray, Y_Test:np.ndarray) -> None:

    '''
    Plots the Model's ROC Curve:
    FitModel := Trained AdaBoost Classifier 
    X_Test := Array with the Feature's Test set  
    Y_Test := Array with the Label's Test set
    '''

    # Predict Probability of belonging to a certain class
    Y_Pred_Proba = FitModel.predict_proba(X_Test)[::,1]
    
    # Getting the ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(Y_Test, Y_Pred_Proba)
    
    # Calculating the Area Under Curve
    AUC = roc_auc_score(Y_Test, Y_Pred_Proba)

    # Adds an Axes to the current figure 
    ax = plt.subplot()
    
    # Setting the Title
    ax.set_title('ROC Curve')

    # Setting the names to the X & Y labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Plotting the Results
    plt.plot(false_positive_rate,
             true_positive_rate,
             label=f"AUC = {round(AUC, 4)}",
             color="darkblue",
             linestyle='-',
             linewidth=1.4)
    
    # Creating the y=x function into the graph
    ax.axline((0, 0), slope=1, label="Chance Level (AUC = 0.5)", color="darkred", linestyle='--')
    
    plt.legend(loc=4)
    plt.show()

def Plot_Weak_Learners_Stats(FitModel:AdaBoost) -> None:

    '''
    Plots The Weak Learner's Training Error and Weights over N Boosting Rounds
    FitModel := Trained AdaBoost Classifier 
    '''

    weak_learners_info = pd.DataFrame({
            "Number of Weak Learners": range(1, len(FitModel.alphas) + 1),
            "Errors": FitModel.training_errors,
            "Weights": FitModel.alphas,
        }
    ).set_index("Number of Weak Learners")
    
    axs = weak_learners_info.plot(
        subplots=True, layout=(1, 2), figsize=(10, 4), legend=False, color="tab:blue"
    )
    axs[0, 0].set_ylabel("Train error")
    axs[0, 0].set_title("Weak learner's training error")
    axs[0, 0].hlines(0.5, 0, 400, colors = '#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
    axs[0, 0].legend()
    
    axs[0, 1].set_ylabel("Weight")
    axs[0, 1].set_title("Weak learner's weight")
    axs[0, 1].legend()

    fig = axs[0, 0].get_figure()
    fig.suptitle("AdaBoostClassifier [Performance Analysis]")
    fig.tight_layout()

def Plot_Model_Stats(FitModel:AdaBoost, X_Test:np.ndarray, Y_Test:np.ndarray, Title:str="Model Performance Evaluation") -> None:

    '''
    Plots The Model's weak learner's Training Error and Weights over N Boosting Rounds as well as the ROC Curve and the Confusion Matrix
    FitModel := Trained AdaBoost Classifier 
    X_Test := Array with the Feature's Test set  
    Y_Test := Array with the Label's Test set
    '''

    # Predict Probability of belonging to a certain class
    Y_Pred_Proba = FitModel.predict_proba(X_Test)[:,1]

    # Getting the ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(Y_Test, Y_Pred_Proba)

    # Calculating the Area Under Curve
    AUC = roc_auc_score(Y_Test, Y_Pred_Proba)

    # Prepare DataFrame for the weak learners' statistics
    weak_learners_info = pd.DataFrame({
            "Errors": FitModel.training_errors,
            "Weights": FitModel.alphas,
        },
        index=range(1, len(FitModel.alphas) + 1)
    )

    # Create a larger figure to accommodate the plots
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))  # adjust the figure size as needed

    # Plot training errors and weights
    weak_learners_info['Errors'].plot(ax=axs[0], title="Weak learner's training error", color="tab:blue")
    axs[0].set_xlabel("# Weak Learners")
    axs[0].set_ylabel("Train error")
    axs[0].hlines(0.5, 1, len(FitModel.alphas), colors='#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
    axs[0].legend()

    weak_learners_info['Weights'].plot(ax=axs[1], title="Weak learner's weight", color="tab:blue")
    axs[1].set_xlabel("# Weak Learners")
    axs[1].set_ylabel("Weight")
    axs[1].legend()

    # Creating a Confusion Matrix
    cm = confusion_matrix(Y_Test, FitModel.predict(X_Test))

    # Creating a HeatMap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axs[2])

    # Plot Confusion Matrix
    axs[2].set_title('Confusion Matrix')
    axs[2].set_xlabel('Predicted Labels')
    axs[2].set_ylabel('True Labels')
    axs[2].xaxis.set_ticklabels(np.unique(Y_Test))
    axs[2].yaxis.set_ticklabels(np.unique(Y_Test))

    # Plot ROC Curve
    axs[3].plot(false_positive_rate, true_positive_rate, label=f"AUC = {round(AUC, 4)}", color="darkblue", linestyle='-', linewidth=1.4)
    axs[3].plot([0, 1], [0, 1], label="Chance level (AUC = 0.5)", color="darkred", linestyle='--')
    axs[3].set_title('ROC Curve')
    axs[3].set_xlabel('False Positive Rate')
    axs[3].set_ylabel('True Positive Rate')
    axs[3].legend()

    # Set the super title for all subplots
    fig.suptitle(Title)

    plt.tight_layout()
    plt.show()

def Plot_Average_Model_Stats(average_results, model_name, Title:str="Model Performance Evaluation") -> None:

    # Todos os modelos treinados
    # Cada um dos test sets

    '''
    Plots The Average Model's weak learner's Training Error and Weights over N Boosting Rounds as well as the ROC Curve
    FitModel := Trained AdaBoost Classifier 
    X_Test := Array with the Feature's Test set  
    Y_Test := Array with the Label's Test set
    '''

    
    # Calculating the Mean Values
    false_positive_rate = average_results[model_name]['fpr']
    true_positive_rate = average_results[model_name]['tpr']
    AUC = average_results[model_name]['AUC']
    train_errors = average_results[model_name]['train_errors']
    alphas = average_results[model_name]['alphas']

    # Prepare DataFrame for the weak learners' statistics
    weak_learners_info = pd.DataFrame({
            "Errors": train_errors,
            "Weights": alphas,
        },
        index=range(1, len(alphas) + 1)
    )

    # Create a larger figure to accommodate the plots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Plot training errors and weights
    weak_learners_info['Errors'].plot(ax=axs[0], title="Weak learner's training error", color="tab:blue")
    axs[0].set_xlabel("# Weak Learners")
    axs[0].set_ylabel("Train error")
    axs[0].hlines(0.5, 1, len(alphas), colors='#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
    axs[0].legend()

    weak_learners_info['Weights'].plot(ax=axs[1], title="Weak learner's weight", color="tab:blue")
    axs[1].set_xlabel("# Weak Learners")
    axs[1].set_ylabel("Weight")
    axs[1].legend()

    # Plot ROC Curve
    axs[2].plot(false_positive_rate, true_positive_rate, label=f"AUC = {round(AUC, 4)}", color="darkblue", linestyle='-', linewidth=1.4)
    axs[2].plot([0, 1], [0, 1], label="Chance level (AUC = 0.5)", color="darkred", linestyle='--')
    axs[2].set_title('ROC Curve')
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].legend()

    # Set the super title for all subplots
    fig.suptitle(Title)

    plt.tight_layout()
    plt.show()

def Compare_Models_Stats(FitModels:list[AdaBoost], ModelsNames:list[str], X_test:np.ndarray, y_test:np.ndarray) -> None:

    '''
    Plots The Model's weak learner's Training Error and Weights over N Boosting Rounds 
    as well as the ROC Curve and the Confusion Matrix
    
    FitModels := List with Trained AdaBoost Classifiers
    ModelsNames := List with the names of the Classifiers
    X_test := Array with the Feature's Test set  
    y_test := Array with the Label's Test set
    '''

    # Create the figure and axes
    fig, axes = plt.subplots(nrows=len(FitModels), ncols=4, figsize=(16, 3.5*len(FitModels)))
    
    for idx, model in enumerate(FitModels):
        # Get Probability of belonging to a certain class
        Y_Pred_Proba = model.predict_proba(X_test)[:,1]
    
        # Get the ROC Curve
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, Y_Pred_Proba)
    
        # Calculate the AUC
        AUC = roc_auc_score(y_test, Y_Pred_Proba)
    
        # Create DataFrame for the weak learners' statistics
        weak_learners_info = pd.DataFrame({
            "Errors": model.training_errors,
            "Weights": model.alphas,
        }, index=range(1, len(model.alphas) + 1))
    
        # Plot training errors
        weak_learners_info['Errors'].plot(ax=axes[idx, 0], title=f"{ModelsNames[idx]} Training Errors", color="tab:blue")
        axes[idx, 0].set_xlabel("# Weak Learners")
        axes[idx, 0].set_ylabel("Training Error")
        axes[idx, 0].hlines(0.5, 1, len(model.alphas), colors='#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
        axes[idx, 0].legend()
    
        # Plot weights
        weak_learners_info['Weights'].plot(ax=axes[idx, 1], title=f"{ModelsNames[idx]} Weights", color="tab:blue")
        axes[idx, 1].set_xlabel("# Weak Learners")
        axes[idx, 1].set_ylabel("Weights")
        axes[idx, 1].legend()
    
        # Creating a Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))

        # Creating a HeatMap
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[idx, 2])
    
        # Plot Confusion Matrix
        axes[idx, 2].set_title(f'{ModelsNames[idx]} Confusion Matrix')
        axes[idx, 2].set_xlabel('Predicted Labels')
        axes[idx, 2].set_ylabel('True Labels')
        axes[idx, 2].xaxis.set_ticklabels(np.unique(y_test))
        axes[idx, 2].yaxis.set_ticklabels(np.unique(y_test))

        # Plot ROC Curve
        axes[idx, 3].plot(false_positive_rate, true_positive_rate, label=f"AUC = {round(AUC, 4)}", color="darkblue", linestyle='-', linewidth=1.4)
        axes[idx, 3].plot([0, 1], [0, 1], color="darkred", linestyle='--')
        axes[idx, 3].axline((0, 0), slope=1, label="Chance Level (AUC = 0.5)", color="darkred", linestyle='--')
        axes[idx, 3].set_title(f'{ModelsNames[idx]} ROC Curve')
        axes[idx, 3].set_xlabel('False Positive Rate')
        axes[idx, 3].set_ylabel('True Positive Rate')
        axes[idx, 3].legend()
    
    # Set the super title for all subplots
    fig.suptitle("Model's Performance Evaluation")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def Compare_Models_Accuracies(X:np.ndarray, y:np.ndarray, Models:list[AdaBoost], ModelsNames:list[str], Colors:dict) -> None:

    '''
    Plots how the Accuracies of the provided Models evolve during K-Fold Cross Validation.
    X := Features Array
    y := Target Array
    Models := List with all the Models to analyse
    ModelsNames := List with the Models Names
    Colors := Colors to use in plotting
    '''

    # Verify the varibles sizes
    assert(len(Models) == len(ModelsNames) and len(ModelsNames) == len(Colors))

    # Create a figure to plot into
    plt.figure()

    for idx, model in enumerate(Models):
        # Calculate Model's Accuracies
        Accuracies, _ = Perform_KFold_CV(X, y, model, 20)
    
        # Plot each model's accuracies
        plt.plot(Accuracies, label=ModelsNames[idx], marker='o', linestyle='-', color=Colors[idx])

    plt.xticks(range(len(Accuracies)), range(1, len(Accuracies) + 1))
    
    # Add labels and title
    plt.xlabel('Number of Folds')
    plt.ylabel('Accuracy')
    plt.title('Models Accuracies')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()

def Model_Accuracies_Per_Dataset(results:pd.DataFrame, model_name:str) -> None:

    '''
    Plots a Barplot with the Model's Accuracies per Dataset previously calculated - Inside Results
    results := dataframe with the accuracies obtained by the model
    model_name := model's name which must coincide with the respective column in the dataframe
    '''

    # Confirm that the column exists
    assert(model_name in results.columns)

    # Asserting the data
    datasets = results['Dataset']
    accuracies = results[model_name]

    # Create the main canvas
    plt.figure(figsize=(12, 8))
    bar_height = 0.4

    # Create the Barplot
    plt.barh(datasets, accuracies, height=bar_height, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title(f'{model_name} Accuracy per Dataset')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def Plot_Critial_Difference_Diagram(Matrix:np.ndarray, Colors:dict) -> None:

    '''
    Plots the Critical Difference Diagram.
    Matrix := Dataframe with the Accuracies obtained by the Models
    Colors := Dictionary that matches each column of the df to a color to use in the Diagram
    '''
    
    # Calculate ranks
    ranks = Matrix.rank(axis=1, ascending=False).mean()
    
    # Perform Nemenyi post-hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(Matrix)

    # Add Some Styling
    marker = {'marker':'o', 'linewidth':1}
    label_props = {'backgroundcolor':'#ADD5F7', 'verticalalignment':'top'}
    
    # Plot the Critical Difference Diagram
    _ = sp.critical_difference_diagram(ranks, nemenyi, color_palette=Colors, marker_props=marker, label_props=label_props)