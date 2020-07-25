
import logging
from sklearn.ensemble import RandomForestClassifier
from src.config import Consts as Cs
import os
import pickle


class ModelCreator:
    """
    A versatile model class for classification purpose
    """
    def __init__(self, model=None, parameters=None):
        """
        create an empty model

        :param model: a Machine Learning Model, if None use RandomForest
        :param parameters: parameters that can be passed to the Model
        """
        if parameters is None:
            parameters = {
                'max_depth': 15,
                'n_estimators': 200,
                'n_jobs': 6}
        self.model_parameters = parameters
        if model is None:
            model = RandomForestClassifier(**self.model_parameters)
        self.model = model

    def save_model(self, filename):
        """
        save Model instance to a pickle

        :param filename: filename
        """
        path_save = os.path.join(Cs.PATH_SAVE_MODEL, filename)
        file = open(path_save, 'ab')
        pickle.dump(self.model, file)
        file.close()

    def load_model(self, filename):
        """
        load a previously trained model

        :param filename: filename
        """
        path_load = os.path.join(Cs.PATH_SAVE_MODEL, filename)
        file = open(path_load, 'rb')
        self.model = pickle.load(file)
        file.close()

    def train(self, x, y):
        """
        fit the classifier to the data

        :param x: a pandas dataframe or numpy array with the training set
        :param y: the labels for the classifier
        """
        self.model.fit(x, y)
        logging.info(f"{self.model.__str__()} fitted")

    def predict(self, x):
        """
        predict the probability of loan default for each client

        :param x: a pandas dataframe with customers features
        :return: the predicted prob of default
        """
        pred = self.model.predict_proba(x)[:, 1]
        logging.info("Default Risk Predicted")

        return pred

