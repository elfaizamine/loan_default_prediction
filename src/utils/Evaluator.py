from matplotlib import pyplot as plt
from sklearn import metrics


class Evaluator:

    def __init__(self, y_true, y_pred):
        """
        calculate auc : area under roc curve
        """
        self.auc = self.auc(y_true, y_pred)

    def auc(self, y_true, y_pred):

        FF, TT, thresholds = metrics.roc_curve(y_true, y_pred)
        self.auc = metrics.auc(FF, TT)


