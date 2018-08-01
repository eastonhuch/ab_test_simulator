import current_approach
import evaluation_functions
import pickle

def most_likely(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    P = ALPHA / (ALPHA + BETA) # Baseline conversion rate
    D = 0.01 # Minimum detectable effect
    TYPE_1 = 0.05
    TYPE_2  = 0.2
    N = current_approach.power_calculation(TYPE_1, TYPE_2, P, D)
    if N < max_test_size:
        max_test_size = N
    
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        N = A_SUCCESS + A_FAIL + B_SUCCESS + B_FAIL
        if N >= max_test_size:
            P_A = A_SUCCESS / (A_SUCCESS + A_FAIL)
            P_B = B_SUCCESS / (B_SUCCESS + B_FAIL)
            decision_dict['ESTIMATED_DIFFERENCE'] = P_B - P_A
            if P_B > P_A:
                decision_dict['DECISION'] = 'B'
            else:
                decision_dict['DECISION'] = 'A'
        return decision_dict

    return decision_function

DATA_FILE = 'test_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)
evaluation_functions.evaluate_all(data_dict, most_likely)