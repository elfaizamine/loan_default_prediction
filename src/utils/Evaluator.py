from Utils import plot_cumulative_gain, cumulative_gain_curve
from matplotlib import pyplot as plt
from sklearn import metrics


class Evaluator:

    def __init__(self):
        """
        This object deals with all data loading tasks
        Args:

        """

        # Master data files & static files
        self.path_data_static = 'path'

    def get_lift_factor_at_percentage(self, a, b, list_percentage):
        list_res = list()
        for i in list_percentage:
            nb_clients = int(len(a) * i / 100)
            list_res.append(b[nb_clients] / a[nb_clients])
        return list_res

    def show_results(self, test):
        """
        Show different KPIs based a on predicted dataframe
        :param test: dataframe with
        :return:
        """
        plot_cumulative_gain(test.TARGET, test.PREDICTION)

        fpr, tpr, _ = metrics.roc_curve(test.TARGET, test.PREDICTION)
        auc = metrics.roc_auc_score(test.TARGET, test.PREDICTION)
        plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()

        a, b = cumulative_gain_curve(test.TARGET, test.PREDICTION)

        list_percentage = [0.1, 1, 5, 10, 20]

        list_res = self.get_lift_factor_at_percentage(a, b, list_percentage)
        for i in range(len(list_res)):
            print("for "+str(list_percentage[i])+"% of the base ("+str(int(5000000*list_percentage[i]/100)) +
                  ' clients), the lift factor is '+str(list_res[i]))

