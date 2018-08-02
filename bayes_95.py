import numpy as np
from evaluation_functions import evaluate_all
from generate_data import get_mme, read_data
from bayes_helpers import sample_posterior

def bayes_95(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    # Number of posterior samples drawn per arm
    N_SAMPLES = 100
    # If the probability of this test being a win is greater than 0.95
    # or the probability of this test being a loss is greater than 0.95
    # Then we stop the test
    P_THRESHOLD = 0.95

    A_MEAN = ALPHA / (ALPHA + BETA)
    A_VAR = (ALPHA * BETA) / ((ALPHA + BETA)**2) / (ALPHA + BETA + 1)
    A_STDDEV = np.sqrt(A_VAR)
    
    B_MEAN = A_MEAN
    B_STDDEV = A_STDDEV * 10
    B_PRIORS = get_mme(B_MEAN, B_STDDEV)
    
    print('ALPHA_A: ' + str(ALPHA))
    print('BETA_A: ' + str(BETA))
    print('ALPHA_B: ' + str(B_PRIORS[0]))
    print('BETA_B: ' + str(B_PRIORS[1]))
    print('Probability Threshold: ' + str(P_THRESHOLD))

    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        A_ALPHA_POST = ALPHA + A_SUCCESS
        A_BETA_POST  = BETA  + B_FAIL
        B_ALPHA_POST = B_PRIORS[0] + B_SUCCESS
        B_BETA_POST  = B_PRIORS[1] + B_FAIL
        N_A = A_SUCCESS + A_FAIL
        N_B = B_SUCCESS + B_FAIL
        N_BOTH = N_A + N_B
        POSTERIOR = sample_posterior(A_ALPHA_POST, A_BETA_POST, 
                                       B_ALPHA_POST, B_BETA_POST,
                                       N_A, N_B, HORIZON_LENGTH - N_BOTH,
                                       N_SAMPLES)
        if POSTERIOR['P_WIN'] > P_THRESHOLD or POSTERIOR['P_WIN'] < (1 - P_THRESHOLD) or N_BOTH >= MAX_TEST_SIZE:
            decision_dict['ESTIMATED_DIFFERENCE'] = POSTERIOR['P_DIFF'] 
            LOSS_DIFF = POSTERIOR['LOSS_B'] - POSTERIOR['LOSS_A']
            if LOSS_DIFF > 0: # Loss for B is greater than loss for A
                decision_dict['DECISION'] = 'A'
            else:
                decision_dict['DECISION'] = 'B'
        return decision_dict

    return decision_function

DATA_FILE = 'sample_data.pkl'
data_dict = read_data(DATA_FILE)
evaluate_all(data_dict, bayes_95)