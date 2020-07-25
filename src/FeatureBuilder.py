from functools import reduce
from src.features.application_loan import *
from src.features.previous_application_loans import *
from src.features.previous_loans_payments import *
import src.config.Consts as cs
from src.utils.utils import *


class FeatureBuilder:
    """
    Create Table containing features developed from all datafiles and all merged on SK_ID_CURR : (Current application
    loan ID)
    """

    def __init__(self, data):
        """
        initiate input : data and output : masterTable

        :param data: data loader containing all clean tables
        """
        self.dataLoader = data
        self.MasterTable = None

    def create_all_features(self):
        """
        Creates all features and merging by SK_ID_CURR
        """

        # Reference to merge on multiple data frames

        df0 = self.dataLoader.df_loan_app.loc[:, ['SK_ID_CURR', 'DATA_PART']]

        #  Table Current application loan

        df1 = transform_categorical_to_numeric(self.dataLoader.df_loan_app)

        df2 = aggregate_columns_on_col(self.dataLoader.df_loan_app, cs.col_to_aggregate_loan_app)

        df3 = columns_not_changed(self.dataLoader.df_loan_app, cs.col_to_keep_loan_app)

        # previous Application loans Home Credit

        df4 = sum_one_hot_columns_on_rows(self.dataLoader.df_previous_loans_app, cs.previous_app_col_to_aggregate)

        df5 = sum_amount_previous_loan(self.dataLoader.df_previous_loans_app)

        df6 = number_past_loans(self.dataLoader.df_previous_loans_app)

        df7 = max_past_annuity(self.dataLoader.df_previous_loans_app)

        df8 = min_max_past_loans_subscription(self.dataLoader.df_previous_loans_app)

        df9 = min_max_past_loans_duration(self.dataLoader.df_previous_loans_app)

        # Previous application loans payments Home Credit

        df10 = past_loans_duration_type(self.dataLoader.df_previous_loans_payments)

        df11 = number_of_payments(self.dataLoader.df_previous_loans_payments, 'last')

        df12 = number_of_payments(self.dataLoader.df_previous_loans_payments, 'all')

        df13 = payments_amounts_difference_previous_loans(self.dataLoader.df_previous_loans_payments, 'last')

        df14 = payments_amounts_difference_previous_loans(self.dataLoader.df_previous_loans_payments, 'all')

        df15 = payments_time_difference_previous_loans(self.dataLoader.df_previous_loans_payments, 'last')

        df16 = payments_time_difference_previous_loans(self.dataLoader.df_previous_loans_payments, 'all')

        df17 = count_missing_payments_infos(self.dataLoader.df_previous_loans_payments)

        # 1.b Creates target
        target = self.dataLoader.df_loan_app.loc[:, ['SK_ID_CURR', 'TARGET']]

        dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15,
               df16, df17, target]

        # merge features and target
        df_final = reduce(lambda left, right: pd.merge(left, right, on=['SK_ID_CURR'], how='left'), dfs)

        # fill only NA in numeric columns
        col_numeric = df_final.select_dtypes(include=np.number).columns.tolist()
        df_final.loc[:, col_numeric] = df_final.loc[:, col_numeric].fillna(-10)

        self.MasterTable = df_final
