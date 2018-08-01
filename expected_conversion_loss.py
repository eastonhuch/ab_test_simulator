import pickle
import generate_data as gd
import numpy as np
import evaluation_functions
import bayes_helpers as b

def expected_conversion_loss(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    JUMP_ = 20
    NUM_TESTS = 1000 # Used to choose optimal threshold
    N_SAMPLES = 500 # Number of posterior samples drawn per arm
    A_MEAN = ALPHA / (ALPHA + BETA)
    A_VAR = (ALPHA * BETA) / ((ALPHA + BETA)**2) / (ALPHA + BETA + 1)
    A_STDDEV = np.sqrt(A_VAR)
    
    B_MEAN = A_MEAN
    B_STDDEV = A_STDDEV * 10
    B_PRIORS = gd.get_mme(B_MEAN, B_STDDEV)
    
    print('ALPHA_A: ' + str(ALPHA))
    print('BETA_A: ' + str(BETA))
    print('ALPHA_B: ' + str(B_PRIORS[0]))
    print('BETA_B: ' + str(B_PRIORS[1]))



    THRESHOLD = b.get_threshold(ALPHA, BETA, B_PRIORS[0], B_PRIORS[1],
                              MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS,
                              0, HORIZON_LENGTH * A_STDDEV * 25, 51,
                              N_SAMPLES, JUMP=JUMP_)

    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        A_ALPHA_POST = ALPHA + A_SUCCESS
        A_BETA_POST  = BETA  + B_FAIL
        B_ALPHA_POST = B_PRIORS[0] + B_SUCCESS
        B_BETA_POST  = B_PRIORS[1] + B_FAIL
        N_A = A_SUCCESS + A_FAIL
        N_B = B_SUCCESS + B_FAIL
        N_BOTH = N_A + N_B
        POSTERIOR = b.sample_posterior(A_ALPHA_POST, A_BETA_POST, 
                                       B_ALPHA_POST, B_BETA_POST,
                                       N_A, N_B, HORIZON_LENGTH - N_BOTH,
                                       N_SAMPLES)
        LOSS_DIFF = POSTERIOR['LOSS_B'] - POSTERIOR['LOSS_A']
        if abs(LOSS_DIFF) > THRESHOLD or N_BOTH >= MAX_TEST_SIZE:
            decision_dict['ESTIMATED_DIFFERENCE'] = POSTERIOR['P_DIFF'] 
            if LOSS_DIFF > 0: # Loss for B is greater than loss for A
                decision_dict['DECISION'] = 'A'
            else:
                decision_dict['DECISION'] = 'B'
        return decision_dict

    return decision_function

DATA_FILE = 'test_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)

evaluation_functions.evaluate_all(data_dict, expected_conversion_loss)
