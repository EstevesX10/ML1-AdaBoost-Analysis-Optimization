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

'''

def Display_Confusion_Matrix(fit_model, X_Test, y_Test, labels):    
    # Creating a Confusion Matrix
    cm = confusion_matrix(y_Test, fit_model.predict(X_Test))
    
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

def Plot_ROC_Curve(fit_model, X_Test, Y_Test):
    # Predict Probability of belonging to a certain class
    Y_Pred_Proba = fit_model.predict_proba(X_Test)[::,1]
    
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
             label=f"AUC = {AUC:1.3f}",
             color="darkblue",
             linestyle='-',
             linewidth=1.4)
    
    # Creating the y=x function into the graph
    ax.axline((0, 0), slope=1, label="Chance Level (AUC = 0.5)", color="darkred", linestyle='--')
    
    plt.legend(loc=4)
    plt.show()