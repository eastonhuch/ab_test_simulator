import pickle
from scipy.stats import norm
from evaluation_functions import evaluate_all

def power_calculation(TYPE_1, TYPE_2, P, D):
    Z_A = norm.ppf(1 - TYPE_1/2.0)
    Z_B = norm.ppf(TYPE_2)
    N   = int((((Z_A - Z_B) / D)**2) * (P * (1 - P)) * 4) + 1
    return N

def get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
    N_A = A_SUCCESS + A_FAIL
    N_B = B_SUCCESS + B_FAIL
    P_A = A_SUCCESS / N_A
    P_B = B_SUCCESS / N_B
    P_DIFF = P_B - P_A
    S2 = P_A * (1 - P_A) / N_A + P_B * (1 - P_B) / N_B
    S = S2 ** (0.5)
    Z = P_DIFF / S
    return (Z, P_DIFF)

def current_approach(ALPHA, BETA, HORIZON_LENGTH, max_test_size):
    P = ALPHA / (ALPHA + BETA) # Baseline conversion rate
    D = 0.01 # Minimum detectable effect
    TYPE_1 = 0.05
    TYPE_2  = 0.2
    N = power_calculation(TYPE_1, TYPE_2, P, D)
    if N < max_test_size:
        max_test_size = N
    
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        if A_SUCCESS + A_FAIL + B_SUCCESS + B_FAIL >= max_test_size:
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            P_VALUE = (1 - norm.cdf(Z)) * 2
            if P_VALUE < TYPE_1 and Z > 0:
                decision_dict['DECISION'] = 'B'
            else:
                decision_dict['DECISION'] = 'A'
            decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
        return decision_dict

    return decision_function

DATA_FILE = 'test_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)
evaluate_all(data_dict, current_approach)