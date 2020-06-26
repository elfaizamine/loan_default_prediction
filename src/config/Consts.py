# Raw Data Paths And Table Names

RAW_DATA_DIR = "./data/raw/"

TRAIN_LOAN_APPLICATION_FILE_NAME = "application_train.csv"

TEST_LOAN_APPLICATION_FILE_NAME = "application_test.csv"

PREVIOUS_LOAN_APPLICATION_FILE_NAME = "previous_application.csv"

PREVIOUS_LOANS_STATUS_FILE_NAME = "POS_CASH_balance.csv"

OTHER_LOAN_APPLICATION_FILE_NAME = "bureau.csv"

PREVIOUS_LOANS_CREDIT_CARD_BALANCE_FILE_NAME = "credit_card_balance.csv"

PREVIOUS_LOANS_INSTALMENT_PAYMENTS_FILE_NAME = "instalments_payments.csv"

OTHER_PREVIOUS_LOANS_BALANCE_FILE_NAME = "bureau_balance.csv"


# Clean Data Paths And Table Names

CLEAN_DATA_DIR = "./data/clean/"

CLEAN_LOAN_APPLICATION_FILE_NAME = "clean_loan_application.csv"

CLEAN_PREVIOUS_LOANS_APPLICATION_FILE_NAME = "clean_previous_application.csv"


# Independent Variables

keep_10_categories = ['NAME_CASH_LOAN_PURPOSE', 'CODE_REJECT_REASON', 'NAME_YIELD_GROUP']

keep_full_categories = ['WEEKDAY_APPR_PROCESS_START', 'NAME_EDUCATION_TYPE']


# Application Loan Variables

label_enc_loan_app_col = ['WEEKDAY_APPR_PROCESS_START', 'NAME_EDUCATION_TYPE']

label_enc_loan_app_cat = [['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'],
                          ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education',
                           'Academic degree']]

binary_enc_loan_app = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

one_hot_enc_loan_app = ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                        'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'HOUSETYPE_MODE']

col_to_keep_loan_app = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                        'CNT_FAM_MEMBERS' , 'REGION_RATING_CLIENT', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                        'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

col_to_aggregate_loan_app = {
                            'FLAG_PROVIDED_INFO': ['FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE',
                                                   'FLAG_CONT_MOBILE', 'FLAG_PHONE'],
                            'AMT_REQ_CREDIT_BUREAU_MON': ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                                                          'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON'],
                            'AMT_REQ_CREDIT_BUREAU_YEAR': ['AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
                           }


# Previous Application Loan Variables

previous_app_col_to_aggregate = ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'NAME_CLIENT_TYPE',
                                 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE','CHANNEL_TYPE',
                                 'NAME_SELLER_INDUSTRY','PRODUCT_COMBINATION', 'NAME_YIELD_GROUP',
                                 'NAME_CASH_LOAN_PURPOSE']


# Path to save Model :

PATH_SAVE_MODEL = "./models/"

KAGGLE_PATH = "./data/prediction/submission.csv"


