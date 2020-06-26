from functools import reduce
from src.features.application_loan import *
from src.features.previous_application_loans import *
import src.config.Consts as cs
from src.utils.utils import *


class FeatureBuilder:

    def __init__(self, database):

        # database must have loaded the data
        self.database = database

    def create_all_features(self):
        """ Creates all features
        """
        df0 = self.database.df_loan_app.loc[:, ['SK_ID_CURR', 'DATA_PART']]
        
        df1 = transform_categorical_to_numeric(self.database.df_loan_app)
        
        df2 = aggregate_columns(self.database.df_loan_app, cs.col_to_aggregate_loan_app)
        
        df3 = columns_not_changed(self.database.df_loan_app, cs.col_to_keep_loan_app)

        df4 = aggregate_one_hot_enc_columns(self.database.df_previous_loans_app, cs.previous_app_col_to_aggregate)

        df5 = sum_amount_previous_loan(self.database.df_previous_loans_app)

        df6 = number_past_loans(self.database.df_previous_loans_app)

        df7 = max_past_annuity(self.database.df_previous_loans_app)

        df8 = min_max_past_loans_subscription(self.database.df_previous_loans_app)

        df9 = min_max_past_loans_duration(self.database.df_previous_loans_app)

        # 1.b Creates target
        target = self.database.df_loan_app.loc[:, ['SK_ID_CURR', 'TARGET']]

        dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, target]

        df_final = reduce(lambda left, right: pd.merge(left, right, on=['SK_ID_CURR'], how='left'), dfs)

        # fill only NA in numeric columns
        col_numeric = df_final.select_dtypes(include=np.number).columns.tolist()
        df_final.loc[:, col_numeric] = df_final.loc[:, col_numeric].fillna(-100)

        return df_final
