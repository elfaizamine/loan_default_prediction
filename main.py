from src.data import DataLoader
from src import FeatureBuilder
from src.models.Model import ModelCreator
from src.config import Consts as cs
import json
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # Loading Clean Data
    loader = DataLoader.DataLoader()
    loader.load_all()

    # Feature Construction
    feature_creator = FeatureBuilder.FeatureBuilder(loader)
    feature_creator.create_all_features()
    master_table = feature_creator.MasterTable

    # Split Data to Train and Test
    master_table_train = master_table[master_table.DATA_PART == 'train'].drop(['DATA_PART'], axis=1)
    master_table_test = master_table[master_table.DATA_PART == 'test'].drop(['DATA_PART'], axis=1)

    # features_to_eliminate = ['SK_ID_CURR', 'TARGET']
    # features = [x for x in MasterTable_Train.keys() if x not in features_to_eliminate]

    # load optimal features according to feature selector
    with open(cs.PATH_SAVE_BEST_FEATURES, 'rb') as file:
        best_features = json.load(file)

    # Split data to input and output
    x_train = master_table_train[best_features].values
    y_train = master_table_train.TARGET
    x_test = master_table_test[best_features].values

    # Model Training and Prediction
    model = ModelCreator(parameters=cs.TRAIN_PARAMETERS)
    model.train(x_train, y_train)
    master_table_test['PREDICTION'] = model.predict(x_test)

    # Save prediction file at data/prediction to submit on kaggle
    kaggle_submission = master_table_test.loc[:, ['SK_ID_CURR', 'PREDICTION']].rename(columns={'PREDICTION': 'TARGET'})
    kaggle_submission.to_csv(cs.KAGGLE_PATH, index=False)





