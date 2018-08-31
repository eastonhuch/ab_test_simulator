import random
from evaluation_functions import evaluate_all
from generate_data import read_data

def just_one(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': random.choice(['A', 'B']),
                         'ESTIMATED_DIFFERENCE': 0.00}
        return decision_dict
    return decision_function

# For testing
#DATA_FILE = 'sample_data.pkl'
#data_dict = read_data(DATA_FILE)
#evaluate_all(data_dict, just_one)