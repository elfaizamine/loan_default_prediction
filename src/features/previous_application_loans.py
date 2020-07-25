from src.utils.utils import *
import src.config.Consts as cs


def sum_one_hot_columns_on_rows(df, col_to_aggregate):
    """
    aggregate past loans information (category columns) in one row

    :param df: previous loans application dataframe
    :param col_to_aggregate: columns to be aggregated

    :return  dataframe aggregated columns plus SK_ID_CURR
    """
    df = one_hot_encoding(df, col_to_aggregate, ['SK_ID_CURR'], drop_first=False)
    column = df.columns.tolist()
    column.remove('SK_ID_CURR')
    df_out = df.groupby(['SK_ID_CURR'])[column].sum().reset_index()

    return df_out


def sum_amount_previous_loan(df):
    """
    sum previous loans amount (demanded amount and obtained amount)

    :param df: previous loans application dataframe

    :return  dataframe : SK_ID_CURR plus feature PREV_AMT_APP_SUM & PREV_AMT_CRED_SUM
    """
    df_out = df.groupby(['SK_ID_CURR'])['AMT_APPLICATION', 'AMT_CREDIT'].sum().reset_index(). \
        rename(columns={'AMT_APPLICATION': 'PREV_AMT_APP_SUM', 'AMT_CREDIT': 'PREV_AMT_CRED_SUM'})
    return df_out


def number_past_loans(df):
    """
    count number of past loans

    :param df: previous loans application dataframe

    :return  dataframe : SK_ID_CURR plus feature NUMBER_PREVIOUS_LOANS
    """
    df = df.groupby(['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index(). \
        rename(columns={'SK_ID_PREV': 'NUMBER_PREVIOUS_LOANS'})
    return df


def max_past_annuity(df):
    """
    max annuity payed on any previous loans

    :param df: previous loans application dataframe

    :return  dataframe : SK_ID_CURR plus feature MAX_PAST_ANNUITY
    """
    df = df.groupby(['SK_ID_CURR'])['AMT_ANNUITY'].max().reset_index(). \
        rename(columns={'AMT_ANNUITY': 'MAX_PAST_ANNUITY'})
    return df


def min_max_past_loans_subscription(df):
    """
    min max of time relative to actual current loan application

    :param df: previous loans application dataframe

    :return  dataframe : SK_ID_CURR plus feature TIME_SINCE_FIRST_LOAN & TIME_SINCE_LAST_LOAN
    """
    df = df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].agg(['min', 'max']).reset_index(). \
        rename(columns={'min': 'TIME_SINCE_FIRST_LOAN', 'max': 'TIME_SINCE_LAST_LOAN'})
    col_to_turn_positive = ['TIME_SINCE_FIRST_LOAN', 'TIME_SINCE_LAST_LOAN']
    df[col_to_turn_positive] = df[col_to_turn_positive].abs()
    return df


def min_max_past_loans_duration(df):
    """
    min max of past loans duration

    :param df: previous loans application dataframe

    :return  dataframe : SK_ID_CURR plus feature SHORTEST_PAST_LOAN & LONGEST_PAST_LOAN
    """
    df = df.groupby(['SK_ID_CURR'])['CNT_PAYMENT'].agg(['min', 'max']).reset_index(). \
        rename(columns={'min': 'SHORTEST_PAST_LOAN', 'max': 'LONGEST_PAST_LOAN'})
    return df


