import numpy as np
from evaluation_functions import evaluate_all
from generate_data import get_mme, read_data
from bayes_helpers import sample_posterior

def bayes_p(P_THRESHOLD):
    def bayes_inner(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
        N_SAMPLES = 500 # Number of posterior samples drawn per arm
        JUMP = 100 # Evaluate probability every JUMP samples

        # If the probability of this test being a win is greater than 0.95
        # or the probability of this test being a loss is greater than 0.95
        # Then we stop the test
        P_THRESHOLD_ = P_THRESHOLD

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
        print('Probability Threshold: ' + str(P_THRESHOLD_))

        def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
            decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
            N_A = A_SUCCESS + A_FAIL
            N_B = B_SUCCESS + B_FAIL
            N_BOTH = N_A + N_B
            if N_BOTH % JUMP == 0 or N_BOTH > MAX_TEST_SIZE:
                A_ALPHA_POST = ALPHA + A_SUCCESS
                A_BETA_POST  = BETA  + A_FAIL
                B_ALPHA_POST = B_PRIORS[0] + B_SUCCESS
                B_BETA_POST  = B_PRIORS[1] + B_FAIL
                POSTERIOR = sample_posterior(A_ALPHA_POST, A_BETA_POST, 
                                            B_ALPHA_POST, B_BETA_POST,
                                            N_A, N_B, HORIZON_LENGTH - N_BOTH,
                                            N_SAMPLES)
                if POSTERIOR['P_WIN'] > P_THRESHOLD_ or POSTERIOR['P_WIN'] < (1 - P_THRESHOLD_) or N_BOTH >= MAX_TEST_SIZE:
                    decision_dict['ESTIMATED_DIFFERENCE'] = POSTERIOR['P_DIFF'] 
                    if POSTERIOR['P_WIN'] > 0.5: # Loss for B is greater than loss for A
                        decision_dict['DECISION'] = 'B'
                    else:
                        decision_dict['DECISION'] = 'A'
            return decision_dict
        return decision_function
    return bayes_inner

# For testing
#DATA_FILE = 'small_data.pkl'
#data_dict = read_data(DATA_FILE)
#evaluate_all(data_dict, bayes_p(0.95))