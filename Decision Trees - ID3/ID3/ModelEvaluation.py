import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold)
from sklearn.model_selection import (cross_val_score)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score)

class Metrics:
    def __init__(self, y_pred:np.ndarray, y_true:np.ndarray) -> None:
        self.y_pred = y_pred
        self.y_true = y_true
    
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)
    
    def balanced_accuracy(self) -> float:
        return balanced_accuracy_score(self.y_true, self.y_pred)

    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred, average="weighted")

    def recall(self) -> float: # Also known as Sensitivity
        return recall_score(self.y_true, self.y_pred, average="weighted")

    def f1_score(self) -> float:
        return f1_score(self.y_true, self.y_pred, average="weighted")

    def calculate_metrics(self) -> pd.DataFrame:
        cols = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score']
        data = [[self.accuracy(), self.balanced_accuracy(), self.precision(), self.recall(),self.f1_score()]]
        df = pd.DataFrame(data, columns=cols)
        return df

if __name__ == "__main__":
    y_true = np.array([1, 1, -1, -1, -1])
    y_pred = np.array([1, 1, -1, -1, 1])
    m = Metrics(y_pred, y_true)
    print(m.accuracy())