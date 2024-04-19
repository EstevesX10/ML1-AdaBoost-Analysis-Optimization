import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score)

'''
    # ----------------------- #
    | Data Visualization File |
    # ----------------------- #

This File contains multiple functions used to Visualize Data:

    -> Display_Confusion_Matrix(fit_model, X_Test, y_Test, labels):
        - Displays a Confusion Matrix of a given Model
    
    -> def Plot_ROC_Curve(fit_model, X_Test, Y_Test):
        - Plots the ROC Curve

    -> Plot_Weak_Learners_Stats(FitModel):
        - Plots Statistics regarding the Variance of the Weak Learner's Error during Training as well as their weights throughout training
    
    -> Compare_Models_Stats(FitModels, ModelsNames, X_test, y_test):
        - Plots Statiscal Data (Error in Training, Weights and ROC Curve) for a list of given Trained Models - Allow for parallel comparison

'''

def Display_Confusion_Matrix(FitModel, X_Test, y_Test, labels):

    '''
    Displays the Model's Confusion Matrix:
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

def Plot_ROC_Curve(FitModel, X_Test, Y_Test):

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

def Plot_Weak_Learners_Stats(FitModel):

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

def Plot_Model_Stats(FitModel, X_Test, Y_Test):

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
    axs[0].set_xlabel("Number of Weak Learners")
    axs[0].set_ylabel("Train error")
    axs[0].hlines(0.5, 1, len(FitModel.alphas), colors='#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
    axs[0].legend()

    weak_learners_info['Weights'].plot(ax=axs[1], title="Weak learner's weight", color="tab:blue")
    axs[1].set_xlabel("Number of Weak Learners")
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
    fig.suptitle("Model Performance Evaluation")

    plt.tight_layout()
    plt.show()


def Compare_Models_Stats(FitModels, ModelsNames, X_test, y_test):

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
        axes[idx, 0].set_xlabel("Number of Weak Learners")
        axes[idx, 0].set_ylabel("Training Error")
        axes[idx, 0].hlines(0.5, 1, len(model.alphas), colors='#bd162c', linestyles='dashed', label='Random Guessing (0.5)')
        axes[idx, 0].legend()
    
        # Plot weights
        weak_learners_info['Weights'].plot(ax=axes[idx, 1], title=f"{ModelsNames[idx]} Weights", color="tab:blue")
        axes[idx, 1].set_xlabel("Number of Weak Learners")
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