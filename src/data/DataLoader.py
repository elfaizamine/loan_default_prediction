import os
import src.config.Consts as cs
import pickle


class DataLoader:
    """
    load clean data
    """
    def __init__(self):
        self.df_loan_app = None
        self.df_previous_loans_app = None
        self.df_previous_loans_payments = None
        '''
        self.df_past_loans_balance = None
        self.df_past_loans_credit_card_balance = None
        self.df_past_loans_instalments_payments = None
        self.df_other_loan_app = None
        self.df_other_loan_balance = None
        '''

    def load_all(self):
        self.load_loan_app()
        self.load_previous_loans_app()
        self.load_previous_loans_payments()
        ''' 
        self.clean_past_loans_balance()
        self.clean_past_loans_credit_card_balance()
        self.clean_past_loans_instalments_payments()
        self.clean_other_loan_app()
        self.clean_other_loan_balance()
        '''

    def load_loan_app(self):

        path_load = os.path.join(cs.CLEAN_DATA_DIR, cs.CLEAN_LOAN_APPLICATION_FILE_NAME)
        file = open(path_load, 'rb')
        self.df_loan_app = pickle.load(file)
        file.close()

    def load_previous_loans_app(self):

        path_load = os.path.join(cs.CLEAN_DATA_DIR, cs.CLEAN_PREVIOUS_LOANS_APPLICATION_FILE_NAME)
        file = open(path_load, 'rb')
        self.df_previous_loans_app = pickle.load(file)
        file.close()

    def load_previous_loans_payments(self):

        path_load = os.path.join(cs.CLEAN_DATA_DIR, cs.CLEAN_PREVIOUS_LOANS_PAYMENTS_FILE_NAME)
        file = open(path_load, 'rb')
        self.df_previous_loans_payments = pickle.load(file)
        file.close()




