import numpy as np

class Metrics:
    def __init__(self, y_pred, y_true) -> None:
        self.y_pred = y_pred
        self.y_true = y_true

    def TP(self):
        return sum(y_pred[idx] == 1 and y_true[idx] == 1 for idx in range(len(y_true)))
    
    def TN(self):
        return sum(y_pred[idx] == -1 and y_true[idx] == -1 for idx in range(len(y_true)))
    
    def FP(self):
        return sum(y_pred[idx] == 1 and y_true[idx] == -1 for idx in range(len(y_true)))
    
    def FN(self):
        return sum(y_pred[idx] == -1 and y_true[idx] == 1 for idx in range(len(y_true)))
    
    def accuracy(self):
        return (self.TP() + self.TN()) / (self.TP() + self.TN() + self.FP() + self.FN())
    
    def balanced_accuracy(self):
        return (self.recall() + self.specificity()) / 2

    def precision(self):
        return (self.TP()) / (self.TP() + self.FP())

    def recall(self): # Also known as Sensitivity
        return (self.TP()) / (self.TP() + self.FN())
    
    def specificity(self):
        return (self.TN()) / (self.TN() + self.FP())

    def f1_score(self):
        return 2  * (self.precision() * self.recall()) / (self.precision() + self.recall())

    # --------

    def true_positive_rate(self):
        return self.recall()
    
    def false_positive_rate(self):
        return 1 - self.specificity()
    
if __name__ == "__main__":
    y_true = np.array([1, 1, -1, -1, -1])
    y_pred = np.array([1, 1, -1, -1, 1])
    m = Metrics(y_pred, y_true)
    print(m.accuracy())