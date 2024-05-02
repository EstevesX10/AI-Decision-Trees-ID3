import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold)
from sklearn.model_selection import (cross_val_score)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score)

class Metrics:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
    
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self): # Also known as Sensitivity
        return recall_score(self.y_true, self.y_pred)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)

    def calculate_metrics(self):
        cols = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score']
        data = [[self.accuracy(), self.balanced_accuracy(), self.precision(), self.recall(),self.f1_score()]]
        df = pd.DataFrame(data, columns=cols)
        return df

def Perform_KFold_CV(Model, X, y):
    cv = KFold(n_splits=10, random_state=10, shuffle=True)
    Accuracies = cross_val_score(Model, X, y, scoring='accuracy',cv=cv, n_jobs=-1)
    return Accuracies

if __name__ == "__main__":
    y_true = np.array([1, 1, -1, -1, -1])
    y_pred = np.array([1, 1, -1, -1, 1])
    m = Metrics(y_pred, y_true)
    print(m.accuracy())