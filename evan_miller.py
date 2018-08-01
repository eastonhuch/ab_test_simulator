import pickle
import numpy as np
import evaluation_functions

# This one is designed to control false positives
def evan_miller(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    P_BASELINE = ALPHA / (ALPHA + BETA)
    THRESHOLD = 2.25 * np.sqrt(MAX_TEST_SIZE * P_BASELINE)
    
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION':'continue', 'ESTIMATED_DIFFERENCE':None}
        D = B_SUCCESS - A_SUCCESS
        N_A = A_SUCCESS + A_FAIL
        N_B = B_SUCCESS + B_FAIL
        N_BOTH = N_A + N_B
        if N_BOTH >= MAX_TEST_SIZE or abs(D) >= THRESHOLD:
            P_A = A_SUCCESS / N_A
            P_B = B_SUCCESS / N_B
            decision_dict['ESTIMATED_DIFFERENCE'] = P_B - P_A
            if D > THRESHOLD: # D > THRESHOLD if we want to control type I error
                decision_dict['DECISION'] = 'B'
            else:
                decision_dict['DECISION'] = 'A'
        return decision_dict
    
    return decision_function

# This one allows you to stop early if you have strong evidence
# Otherwise, it waits until the end and chooses the one with
# the superior observed conversion rate
def evan_miller_most_likely(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    P_BASELINE = ALPHA / (ALPHA + BETA)
    THRESHOLD = 2.25 * np.sqrt(MAX_TEST_SIZE * P_BASELINE)
    
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION':'continue', 'ESTIMATED_DIFFERENCE':None}
        D = B_SUCCESS - A_SUCCESS
        N_A = A_SUCCESS + A_FAIL
        N_B = B_SUCCESS + B_FAIL
        N_BOTH = N_A + N_B
        if N_BOTH >= MAX_TEST_SIZE or abs(D) >= THRESHOLD:
            P_A = A_SUCCESS / N_A
            P_B = B_SUCCESS / N_B
            decision_dict['ESTIMATED_DIFFERENCE'] = P_B - P_A
            if D > 0:
                decision_dict['DECISION'] = 'B'
            else:
                decision_dict['DECISION'] = 'A'
        return decision_dict
    
    return decision_function

# Obtain average loss for your function
DATA_FILE = 'test_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)
evaluation_functions.evaluate_all(data_dict, evan_miller)
evaluation_functions.evaluate_all(data_dict, evan_miller_most_likely)