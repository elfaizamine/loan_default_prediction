import src.config.Consts as cs
import pandas as pd
import numpy as np
import pickle
import os


class ColumnTypeOptimizer:

    def __init__(self, data):
        self.data_clean = data
        self.column_types = {'int': [np.int8, np.int16, np.int32, np.int64],
                             'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                             'float': [np.float16, np.float32, np.float64]}

    def numeric_column(self, col, key):
        for sub_type in self.column_types[key]:
            type_info = np.iinfo(sub_type) if 'int' in key else np.finfo(sub_type)
            if self.data_clean[col].max() <= type_info.max and self.data_clean[col].min() >= type_info.min:
                self.data_clean[col] = self.data_clean[col].astype(sub_type)
                break

    def categorize_column(self, col):
        categories = self.data_clean[col].value_counts()
        thresh_cat = 4
        if col in cs.keep_full_categories:
            thresh_cat = 30
        elif col in cs.keep_10_categories:
            thresh_cat = 10
        valid_categories = categories.head(thresh_cat).index.tolist()
        self.data_clean.loc[np.invert(self.data_clean[col].isin(valid_categories)), col] = 'other_' + col.lower()
        self.data_clean[col] = self.data_clean[col].astype('category')

    def optimize(self):
        for col in self.data_clean.columns:          
            col_type = self.data_clean[col].dtype
            if np.issubdtype(col_type, np.integer):
                if self.data_clean[col].min() < 0:
                    self.numeric_column(col, 'int')
                else:
                    self.numeric_column(col, 'uint')
            elif np.issubdtype(col_type, np.floating):
                self.numeric_column(col, 'float')
            elif col_type == 'object':
                self.data_clean[col].fillna('MV_' + col.lower(), inplace=True)
                if self.data_clean[col].apply(lambda x: isinstance(x, str)).all():
                    self.categorize_column(col)


class RawDataCleaner:

    def __init__(self):
        self.df_loan_app = None
        self.df_previous_loans_app = None
        '''
        self.df_past_loans_app = None
        self.df_past_loans_credit_card_balance = None
        self.df_past_loans_instalments_payments = None
        self.df_other_loan_app = None
        self.df_other_loan_balance = None
        '''

    def clean_all(self):
        self.clean_loan_app()
        self.clean_previous_loans_app()
        '''
        self.clean_past_loans_app()
        self.clean_past_loans_balance()
        self.clean_past_loans_credit_card_balance()
        self.clean_past_loans_instalments_payments()
        self.clean_other_loan_app()
        self.clean_other_loan_balance()
        '''
        '''
    def save_all(self):
        self.save_train_loan_app()
        self.save_test_loan_app()
        self.save_past_loans_app()
        self.save_past_loans_balance()
        self.save_past_loans_credit_card_balance()
        self.save_past_loans_instalments_payments()
        self.save_other_loan_app()
        self.save_other_loan_balance()
        '''

    def clean_loan_app(self):

        train_path_load = os.path.join(cs.RAW_DATA_DIR, cs.TRAIN_LOAN_APPLICATION_FILE_NAME)
        test_path_load = os.path.join(cs.RAW_DATA_DIR, cs.TEST_LOAN_APPLICATION_FILE_NAME)

        df_train = pd.read_csv(train_path_load)
        df_test = pd.read_csv(test_path_load)

        df_train['DATA_PART'] = 'train'
        df_test['DATA_PART'] = 'test'

        df = pd.concat([df_train, df_test], sort=False)
        df.fillna(np.nan, inplace=True)

        optimizer = ColumnTypeOptimizer(df)
        optimizer.optimize()
        self.df_loan_app = optimizer.data_clean

        path_save = os.path.join(cs.CLEAN_DATA_DIR, cs.CLEAN_LOAN_APPLICATION_FILE_NAME)
        file = open(path_save, 'ab')
        pickle.dump(self.df_loan_app, file)
        file.close()

    def clean_previous_loans_app(self):

        path = os.path.join(cs.RAW_DATA_DIR, cs.PREVIOUS_LOAN_APPLICATION_FILE_NAME)
        df = pd.read_csv(path)
        df.fillna(np.nan, inplace=True)

        optimizer = ColumnTypeOptimizer(df)
        optimizer.optimize()
        self.df_previous_loans_app = optimizer.data_clean

        path_save = os.path.join(cs.CLEAN_DATA_DIR, cs.CLEAN_PREVIOUS_LOANS_APPLICATION_FILE_NAME)
        file = open(path_save, 'ab')
        pickle.dump(self.df_previous_loans_app, file)
        file.close()
