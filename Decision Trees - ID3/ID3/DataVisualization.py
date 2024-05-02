from IPython.display import (HTML, display)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import (confusion_matrix)

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

def calc_learning_curve_points(Model, X, y, n_itr=50, min_train_samples=1):
    points = []
    n_samples = len(X)
    max_test_size = (n_samples - min_train_samples) / n_samples
    test_sizes = np.linspace(min_train_samples / n_samples, max_test_size, 25)
    
    for test_size in test_sizes:
        accuracies = []
        for _ in range(n_itr):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=13
            )
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

    # Create a larger figure to accommodate the plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

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


    # Unpacking the points into x and y coordinates
    x, y = zip(*Points)
    
    # Create the scatter plot
    axs[1].scatter(x, y, color='#1F7799')
    
    # Using LOWESS to fit a smooth line through the points
    lowess = sm.nonparametric.lowess(y, x, frac=0.2)  # Adjust frac to change the smoothness
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # Plotting the LOWESS result
    axs[1].plot(lowess_x, lowess_y, '#AF1021', linestyle='--')
    
    # Adding titles and labels
    axs[1].set_title('Learning Curve')
    axs[1].set_xlabel('Training Set Size (%)')
    axs[1].set_ylabel('Accuracy on Test Set')

    # Set the super title for all subplots
    fig.suptitle(Title)

    plt.tight_layout()
    plt.show()
