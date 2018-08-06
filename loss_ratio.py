import numpy as np
from generate_data import get_mme, read_data
from evaluation_functions import evaluate_all
from bayes_helpers import sample_posterior, get_threshold

def loss_ratio(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    JUMP_ = 100
    NUM_TESTS = 4000 # Used to choose optimal threshold
    N_SAMPLES = 500 # Number of posterior samples drawn per arm
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

    THRESHOLD = get_threshold(ALPHA, BETA, B_PRIORS[0], B_PRIORS[1],
                              MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS,
                              2, 100, 99, N_SAMPLES, JUMP=JUMP_, RATIO=True)

    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        N_A = A_SUCCESS + A_FAIL
        N_B = B_SUCCESS + B_FAIL
        N_BOTH = N_A + N_B
        if N_BOTH % JUMP_ == 0 or N_BOTH >= MAX_TEST_SIZE:
            A_ALPHA_POST = ALPHA + A_SUCCESS
            A_BETA_POST  = BETA  + A_FAIL
            B_ALPHA_POST = B_PRIORS[0] + B_SUCCESS
            B_BETA_POST  = B_PRIORS[1] + B_FAIL
            POSTERIOR = sample_posterior(A_ALPHA_POST, A_BETA_POST, 
                                        B_ALPHA_POST, B_BETA_POST,
                                        N_A, N_B, HORIZON_LENGTH - N_BOTH,
                                        N_SAMPLES)
            LOSS_RATIO = POSTERIOR['LOSS_B'] / POSTERIOR['LOSS_A']
            if LOSS_RATIO > THRESHOLD or LOSS_RATIO < 1/THRESHOLD or N_BOTH >= MAX_TEST_SIZE:
                decision_dict['ESTIMATED_DIFFERENCE'] = POSTERIOR['P_DIFF'] 
                if LOSS_RATIO > 1: # Loss for B is greater than loss for A
                    decision_dict['DECISION'] = 'A'
                else:
                    decision_dict['DECISION'] = 'B'
        return decision_dict

    return decision_function

# for testing
DATA_FILE = 'test_data.pkl'
data_dict = read_data(DATA_FILE)
evaluate_all(data_dict, loss_ratio)
