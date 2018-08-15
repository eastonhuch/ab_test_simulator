from scipy.stats import norm
from evaluation_functions import evaluate_all
from generate_data import read_data
from current_approach import power_calculation, get_Z

def peek_1(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    P = ALPHA / (ALPHA + BETA) # Baseline conversion rate
    D = 0.01 # Minimum detectable effect
    TYPE_1 = 0.05
    TYPE_2  = 0.2
    N = power_calculation(TYPE_1, TYPE_2, P, D)
    if N < max_test_size:
        max_test_size = N
    MIN_SAMPLES = 100
    print('I\'ll start peeking at ' + str(MIN_SAMPLES) + ' samples')  
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
        elif N >= MIN_SAMPLES:
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            P_VALUE = (1 - norm.cdf(abs(Z))) * 2
            if P_VALUE < TYPE_1:
                decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
                if Z > 0:
                    decision_dict['DECISION'] = 'B'
                else:
                    decision_dict['DECISION'] = 'A'
        return decision_dict
    return decision_function

def peek_100(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    P = ALPHA / (ALPHA + BETA) # Baseline conversion rate
    D = 0.01 # Minimum detectable effect
    TYPE_1 = 0.05
    TYPE_2  = 0.2
    N = power_calculation(TYPE_1, TYPE_2, P, D)
    if N < max_test_size:
        max_test_size = N
    PEEK = 100
    print('I\'m peeking every ' + str(PEEK) + ' samples')  
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
        elif N % PEEK == 0:
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            P_VALUE = (1 - norm.cdf(abs(Z))) * 2
            if P_VALUE < TYPE_1:
                decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
                if Z > 0:
                    decision_dict['DECISION'] = 'B'
                else:
                    decision_dict['DECISION'] = 'A'
        return decision_dict
    return decision_function

def peek_for_loss(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    P = ALPHA / (ALPHA + BETA) # Baseline conversion rate
    D = 0.01 # Minimum detectable effect
    TYPE_1 = 0.05
    TYPE_2  = 0.2
    N = power_calculation(TYPE_1, TYPE_2, P, D)
    if N < max_test_size:
        max_test_size = N
    PEEK = 100
    print('I\'m peeking every ' + str(PEEK) + ' samples') 
    print('If the B-arm is losing and has a p-value below 0.05, I will stop early') 
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
        elif N % PEEK == 0:
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            P_VALUE = (1 - norm.cdf(abs(Z))) * 2
            if P_VALUE < TYPE_1 and Z > 0:
                decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
                decision_dict['DECISION'] = 'B'
        return decision_dict
    return decision_function

# For testing
#DATA_FILE = 'test_data.pkl'
#data_dict = read_data(DATA_FILE)
#evaluate_all(data_dict, peek_1)
#evaluate_all(data_dict, peek_100)
#evaluate_all(data_dict, peek_for_loss)