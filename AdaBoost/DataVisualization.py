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