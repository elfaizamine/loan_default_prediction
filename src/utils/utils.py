import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def one_hot_encoding(df, columns, key, drop_first=True):

    df_dummies = pd.get_dummies(df[columns], drop_first=drop_first)
    df_dummies[key] = df[key]

    return df_dummies


def binary_encoding(df, columns):

    df_binary = df.loc[:, ['SK_ID_CURR'] + columns]
    ordinalencoder_df = preprocessing.OrdinalEncoder()
    df_binary.loc[:, columns] = ordinalencoder_df.fit_transform(df_binary.loc[:, columns]).astype('int8')

    return df_binary


def label_encoding(df, col_to_label, col_categories):

    df_label = df.loc[:, ['SK_ID_CURR'] + col_to_label]
    for index, col in enumerate(col_to_label):
        df_label[col] = df_label[col].apply(lambda x: col_categories[index].index(x)).astype('int8')

    return df_label


def aggregate_columns(df, columns_to_aggregate):

    df_sum_agg = pd.DataFrame(df.SK_ID_CURR, columns=['SK_ID_CURR'])

    for col_name in columns_to_aggregate:
        df_sum_agg[col_name] = df[columns_to_aggregate[col_name]].sum(axis=1)

    return df_sum_agg


def columns_not_changed(df, col_to_keep):
    df = df.loc[:, ['SK_ID_CURR'] + col_to_keep]
    df.loc[df.DAYS_ID_PUBLISH > 0, :] = np.nan
    col_to_turn_positive = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
    df[col_to_turn_positive] = df[col_to_turn_positive].abs()

    return df
