from src.data import DataLoader
from src import FeatureBuilder
from src.models.Model import ModelCreator
from sklearn import model_selection


class LoanDefaultRisk:
    """
    load data and setting cross validation, model, and metric evaluation parameters
    """

    def __init__(self, method, parameters):
        """
        :param randomseed: seed to ensure same replicated results
        :param parameters: genetic algorithm parameters plus model training parameters

        """
        self.features = None
        self.X = None
        self.y = None
        self.classifier = None
        self.kfold = None

        loader = DataLoader.DataLoader()
        loader.load_all()

        feature_creator = FeatureBuilder.FeatureBuilder(loader)
        feature_creator.create_all_features()
        dataset = feature_creator.MasterTable
        master_table = dataset[dataset.DATA_PART == 'train'].drop(['DATA_PART'], axis=1)

        features = ['SK_ID_CURR', 'TARGET']
        self.features = [x for x in master_table.keys() if x not in features]

        self.X = master_table[self.features]
        self.y = master_table.TARGET

        self.classifier = ModelCreator(parameters=parameters['MODEL']).model

        if method == 'ga':
            self.kfold = model_selection.KFold(n_splits=parameters['CROSS_VALIDATION_FOLDS'],
                                               random_state=parameters['OPTIMISATION']['RANDOM_SEED'])

    def __len__(self):
        """
        :return: the total number of features used in this classification problem

        """
        return self.X.shape[1]

    def getMeanAuc(self, zero_one_list):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList

        :param zero_one_list: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.

        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """
        # drop the dataset columns that correspond to the unselected features:
        zero_indices = [i for i, n in enumerate(zero_one_list) if n == 0]
        currentX = self.X.drop(self.X.columns[zero_indices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold,
                                                     scoring='roc_auc')
        # return mean accuracy:
        return cv_results.mean()