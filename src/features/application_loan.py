from src.utils.utils import *
import src.config.Consts as cs


def transform_categorical_to_numeric(df):
    """
    Transform categorical columns to numeric ones

    :param df: dataframe

    :return  numeric columns plus SK_ID_CURR
    """
    df_label = label_encoding(df, cs.label_enc_loan_app_col, cs.label_enc_loan_app_cat)
    df_binary = binary_encoding(df, cs.binary_enc_loan_app)
    df_one_hot_encoding = one_hot_encoding(df, cs.one_hot_enc_loan_app, ['SK_ID_CURR'])

    df_m1 = df_one_hot_encoding.merge(df_binary, how='left', on=['SK_ID_CURR'])
    df_cat_num = df_label.merge(df_m1, how='left', on=['SK_ID_CURR'])

    return df_cat_num
















