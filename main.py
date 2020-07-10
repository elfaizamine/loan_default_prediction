from src.data import DataLoader
from src import FeatureBuilder
from src.models.Model import ModelCreator
from src.config import Consts as cs
import pandas as pd
import random


if __name__ == '__main__':

    loader = DataLoader.DataLoader()
    loader.load_all()

    feature_creator = FeatureBuilder.FeatureBuilder(loader)
    MasterTable = feature_creator.create_all_features()
    MasterTable_Train = MasterTable[MasterTable.DATA_PART == 'train'].drop(['DATA_PART'], axis=1)
    MasterTable_Test = MasterTable[MasterTable.DATA_PART == 'test'].drop(['DATA_PART'], axis=1)

    features_to_eliminate = ['SK_ID_CURR', 'TARGET']
    features = [x for x in MasterTable_Train.keys() if x not in features_to_eliminate]

    x_train = MasterTable_Train[features].values
    y_train = MasterTable_Train.TARGET
    x_test = MasterTable_Test[features].values

    model = ModelCreator()
    model.train(x_train, y_train)
    MasterTable_Test['PREDICTION'] = model.predict(x_test)

    kaggle_submission = MasterTable_Test.loc[:, ['SK_ID_CURR', 'PREDICTION']]
    kaggle_submission.to_csv(cs.KAGGLE_PATH, index=False)





