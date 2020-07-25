from src.utils.utils import *
import sys


# select last or all past loans payments
def loan_payments_segmentation(df_payments, prefix):
    """
    separate last past loan payments from old loans payments

    :param df_payments: previous loans payments dataframe
    :param prefix: 'last' : selecting only last loan payments, 'all' : selecting all past loans but last loan

    :return dataframe : dataframe payments loans depending on prefix
    """

    df_red = df_payments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT']].drop_duplicates()
    df_last_loan = df_red.groupby(['SK_ID_CURR']).agg({'DAYS_INSTALMENT': ['min', 'max']})
    df_last_loan.columns = ['FIRST_PAYMENT_LOANS_DATE', 'DAYS_INSTALMENT']
    df_last_loan = df_red.merge(df_last_loan, how='inner', on=['SK_ID_CURR', 'DAYS_INSTALMENT']).rename(
        columns={'DAYS_INSTALMENT': 'LAST_PAYMENT_LOANS_DATE'})
    df_last_loan_payments = df_payments.merge(df_last_loan, how='inner', on=['SK_ID_CURR', 'SK_ID_PREV'])

    # return all past loans payments but last
    if prefix == 'all':
        df_result = df_payments[~df_payments.SK_ID_PREV.isin(df_last_loan_payments.SK_ID_PREV)]
    # return last_loan_payments
    elif prefix == 'last':
        df_result = df_last_loan_payments

    return df_result


# extract the distribution statistics of time difference between date of payment and date
def payments_time_difference_previous_loans(df, prefix):
    """
    describe the distribution of payments time difference by (min, max, sum, mean)

    :param df: previous loans payments dataframe
    :param prefix: 'last' : selecting only last loan payments, 'all' : selecting all past loans but last loan

    :return dataframe :SK_ID_CURR plus LATE_PAYMENTS_TIME(min, max, mean, sum), EARLY_PAYMENTS_TIME(min, max, mean, sum)
    """
    df_last = loan_payments_segmentation(df, prefix)
    df_last = df_last[~df_last.DAYS_ENTRY_PAYMENT.isnull()]

    df_last.loc[:, 'PAYMENT_DIFF'] = df_last.loc[:, 'DAYS_ENTRY_PAYMENT'] - df_last.loc[:, 'DAYS_INSTALMENT']
    df_last.loc[:, 'PAYMENT_DIFF'] = df_last.loc[:, 'PAYMENT_DIFF'].astype('float64')
    df_last.loc[:, "LATE_PAYMENTS_TIME"] = df_last.PAYMENT_DIFF.where(df_last.PAYMENT_DIFF > 0, 0)
    df_last.loc[:, "EARLY_PAYMENTS_TIME"] = -1 * df_last.PAYMENT_DIFF.where(df_last.PAYMENT_DIFF < 0, 0)
    df_last_diff = df_last[(df_last.loc[:, "EARLY_PAYMENTS_TIME"] != 0) | (df_last.loc[:, "LATE_PAYMENTS_TIME"] != 0)]

    if prefix == 'all':
        df_last_diff = df_last_diff.groupby(['SK_ID_CURR']).agg({'LATE_PAYMENTS_TIME': ['min', 'max', 'median', 'sum'],
                                                                 'EARLY_PAYMENTS_TIME': ['min', 'max', 'median','sum']})
        df_last_diff.columns = ["_all_".join(x) for x in df_last_diff.columns.ravel()]

    elif prefix == 'last':
        df_last_diff = df_last_diff.groupby(['SK_ID_CURR']).agg({'LATE_PAYMENTS_TIME': ['min', 'max', 'median', 'sum'],
                                                                 'EARLY_PAYMENTS_TIME': ['min', 'max', 'median', 'sum'],
                                                                 'LAST_PAYMENT_LOANS_DATE': ['mean'],
                                                                 'FIRST_PAYMENT_LOANS_DATE': ['mean']})
        df_last_diff.columns = ["_last_".join(x) for x in df_last_diff.columns.ravel()]

    df_last_diff.columns = df_last_diff.columns.str.upper()
    df_last_diff = df_last_diff.reset_index()

    df_last = df_last[['SK_ID_CURR']].drop_duplicates()
    result = df_last.merge(df_last_diff, how='left', on=['SK_ID_CURR'])

    return result


def payments_amounts_difference_previous_loans(df, prefix):
    """
    describe the distribution of payments amount difference by (min, max, sum, mean) for late payments (>=20 days)

    :param df: previous loans payments dataframe
    :param prefix: 'last' : selecting only last loan payments, 'all' : selecting all past loans but last loan

    :return  dataframe :  SK_ID_CURR plus PAYMENT_AMOUNT_REST(min, max, mean, sum)
    """
    epsilon = sys.float_info.epsilon
    
    df = loan_payments_segmentation(df, prefix)
    df = df.loc[~df.AMT_PAYMENT.isnull()]

    # calculate how many days the payments are late
    df.loc[:, 'PAYMENT_DIFF'] = df.loc[:, 'DAYS_ENTRY_PAYMENT'] - df.loc[:, 'DAYS_INSTALMENT']

    # filter on late payments for more than 20 days
    df_late = df.loc[df.PAYMENT_DIFF >= 20, :]
    null_amount_index_ = df_late.loc[df_late.AMT_INSTALMENT == 0, :].index
    df_late.loc[null_amount_index_, 'AMT_INSTALMENT'] = -1 * df_late.loc[null_amount_index_, 'AMT_PAYMENT'].values

    # percentage of annuity amount that is not payed (0 all annuity amount payed, 1 no payment)
    df_late.loc[:, 'PAYMENT_AMOUNT_REST'] = (df_late.loc[:, 'AMT_PAYMENT'] / (df_late.loc[:, 'AMT_INSTALMENT']+epsilon)).round(2)

    # describe the amount difference by statistics: min, max, mean, sum
    df_res = df_late.groupby(['SK_ID_CURR']).agg({'PAYMENT_AMOUNT_REST': ['min', 'max', 'mean', 'sum']})

    df_res.columns = [prefix + "_" + "_".join(x) for x in df_res.columns.ravel()]
    df_res.columns = df_res.columns.str.upper()
    df_res = df_res.reset_index()

    df = df[['SK_ID_CURR']].drop_duplicates()
    result = df.merge(df_res, how='left', on=['SK_ID_CURR'])
    result.fillna(0, inplace=True)

    return result


def number_of_payments(df_payment, prefix):
    """
    count observed number of payments in past loans and predefined number in contract

    :param df_payment: previous loans payments dataframe
    :param prefix: 'last' : selecting only last loan payments, 'all' : selecting all past loans but last loan

    :return  dataframe :  SK_ID_CURR plus NUMBER_ACTUAL_PAYMENTS_(prefix) plus NUMBER_FIXED_PAYMENTS_(prefix)
    """
    df_last = loan_payments_segmentation(df_payment, prefix)
    df_unique = df_last.drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT'])
    df_res1 = df_unique.groupby(['SK_ID_CURR'])['DAYS_ENTRY_PAYMENT'].count().reset_index().rename(
        columns={'DAYS_ENTRY_PAYMENT': 'NUMBER_ACTUAL_PAYMENTS_' + prefix.upper()})
    df_res2 = df_last.groupby(['SK_ID_CURR'])['NUM_INSTALMENT_NUMBER'].max().reset_index().rename(
        columns={'NUM_INSTALMENT_NUMBER': 'NUMBER_FIXED_PAYMENTS_' + prefix.upper()})

    result = df_res1.merge(df_res2, how='outer', on=['SK_ID_CURR'])

    df = df_payment[['SK_ID_CURR']].drop_duplicates()
    result = df.merge(result, how='left', on=['SK_ID_CURR'])
    result.fillna(-1, inplace=True)

    return result


def past_loans_duration_type(df):
    """
    count number of loan duration type on past loans

    :param df: previous loans payments dataframe

    :return  dataframe :  SK_ID_CURR plus sum of binary columns across previous loans (PAST_LOANS_DURATION_TYPE)
    """
    df = df.drop_duplicates(subset=['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_ENTRY_PAYMENT', 'NUM_INSTALMENT_VERSION'])
    df.loc[:, 'NUM_INSTALMENT_VERSION'] = df.loc[:, 'NUM_INSTALMENT_VERSION'].clip(0, 4).astype(int)
    one_hot_feat = one_hot_encoding(df, 'NUM_INSTALMENT_VERSION', ['SK_ID_CURR'], drop_first=False)
    
    result_df = one_hot_feat.loc[:, one_hot_feat.columns != 'SK_ID_CURR'].add_prefix('PAST_LOANS_DURATION_TYPE_')
    
    result_df.loc[:, 'SK_ID_CURR'] = one_hot_feat.loc[:, 'SK_ID_CURR']

    result_df = result_df.groupby(['SK_ID_CURR'])[
        list(set(result_df.columns) - set(['SK_ID_CURR']))].mean().reset_index()
    result_df = result_df.round(2)

    return result_df


def count_missing_payments_infos(df):
    """
    count number of missing values in past loans payments

    :param df: previous loans payments dataframe

    :return  dataframe :  SK_ID_CURR plus NUMBER_MISSING_PAYMENTS_INFO
    """
    df_missing = df.loc[df.DAYS_ENTRY_PAYMENT.isnull(), :]
    df_missing.fillna(1, inplace=True)

    df_missing = df_missing.groupby(['SK_ID_CURR'])['DAYS_ENTRY_PAYMENT'].sum().reset_index().rename(
        columns={'DAYS_ENTRY_PAYMENT': 'NUMBER_MISSING_PAYMENTS_INFO'})

    df = df[['SK_ID_CURR']].drop_duplicates()
    result = df.merge(df_missing, how='left', on=['SK_ID_CURR'])
    result.fillna(0, inplace=True)

    return df

