from IPython.display import (HTML, display)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import (label_binarize)
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, auc)

def Display_dfs_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

def calc_learning_curve_points(Model, X, y, n_itr=20, min_train_samples=1, n_points=10):
    points = []
    n_samples = len(X)
    max_test_size = (n_samples - min_train_samples) / n_samples
    test_sizes = np.linspace(min_train_samples / n_samples, max_test_size, n_points)
    
    for test_size in test_sizes:
        accuracies = []
        for _ in range(n_itr):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)
            model = Model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(sum(y_test ==  y_pred) / len(y_test))
        
        average_accuracy = np.mean(accuracies)
        train_set_size = n_samples - int(n_samples * test_size)
        points.append((train_set_size/n_samples*100, average_accuracy))
    
    return points

def Plot_Model_Stats(FitModel, Points, X_Test, Y_Test, Title="Model Performance Evaluation"):

    '''
    Plots the Confusion Matrix as Well as the Learning Curve
    FitModel := Trained ID3 Model
    Points := List of Points that describe the trade-off between the model's accuracy and the size of the training set
    X_Test := Array with the Feature's Test set  
    Y_Test := Array with the Label's Test set
    '''

    # Calculating the type of Classification Problem (Binary or MultiClass)
    n_labels = len(FitModel.target_values)
    
    # Getting the Target Labels
    labels = FitModel.target_values

    # Create a larger figure to accommodate the plots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Creating a Confusion Matrix
    cm = confusion_matrix(Y_Test, FitModel.predict(X_Test))

    # Creating a HeatMap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axs[0])

    # Plot Confusion Matrix
    axs[0].set_title('Confusion Matrix')
    axs[0].set_xlabel('Predicted Labels')
    axs[0].set_ylabel('True Labels')
    axs[0].xaxis.set_ticklabels(np.unique(Y_Test))
    axs[0].yaxis.set_ticklabels(np.unique(Y_Test))

    # Binary Classification Problem
    if (n_labels == 2):
        # Predict Probability of belonging to a certain class
        Y_Pred_Proba = FitModel.predict_proba(X_Test)[:,1]

        # Getting the ROC Curve
        false_positive_rate, true_positive_rate, _ = roc_curve(Y_Test, Y_Pred_Proba)

        # Calculating the Area Under Curve
        AUC = roc_auc_score(Y_Test, Y_Pred_Proba)

        # Plot ROC Curve
        axs[1].plot(false_positive_rate, true_positive_rate, label=f"AUC = {round(AUC, 4)}", color="darkblue", linestyle='-', linewidth=1.4)
        axs[1].plot([0, 1], [0, 1], label="Chance Level (AUC = 0.5)", color="darkred", linestyle='--')
        axs[1].set_title('ROC Curve')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].legend()

    # MultiClass Classification Problem
    elif (n_labels > 2):
        # Binarize y_test in preparation for ROC calculation
        y_test = label_binarize(Y_Test, classes=labels)
        n_classes = y_test.shape[1]

        # Get probability scores
        y_score = FitModel.predict_proba(X_Test)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves (Note: In this project we work at max with multiclass problems with 3 different classes therefore we only used 3 colors)
        colors = ['#0059b3', '#990000', '#009926']
        for i in range(n_classes):
            axs[1].plot(fpr[i], tpr[i], color=colors[i], lw=1.2, linestyle='--',
                        label=f"AUC of class {labels[i]} = {roc_auc[i]:0.2f})")

        axs[1].plot([0, 1], [0, 1], 'darkblue', linestyle='--', lw=1.4, label=f"Chance Level (AUC = 0.5)")
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title('Multiclass ROC Curve')
        axs[1].legend(loc="lower right", fontsize='small')

    # Unpacking the points into x and y coordinates
    x, y = zip(*Points)
    
    # Create the scatter plot
    axs[2].scatter(x, y, color='#1F7799')
    
    # Using LOWESS to fit a smooth line through the points
    lowess = sm.nonparametric.lowess(y, x, frac=0.2)  # Adjust frac to change the smoothness
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # Plotting the LOWESS result
    axs[2].plot(lowess_x, lowess_y, '#AF1021', linestyle='--')
    
    # Adding titles and labels
    axs[2].set_title('Learning Curve')
    axs[2].set_xlabel('Training Set Size (%)')
    axs[2].set_ylabel('Accuracy on Test Set')

    # Set the super title for all subplots
    fig.suptitle(Title)

    plt.tight_layout()
    plt.show()
