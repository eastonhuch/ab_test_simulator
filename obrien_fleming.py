import pickle
from evaluation_functions import evaluate_all
from current_approach import get_Z

def obrien_fleming(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    THRESHOLDS = [6.9913, 4.8646, 3.9295, 3.367 ,2.9893, 2.7148, 2.504, 2.3358, 2.1975, 2.0811]
    CHECKPOINTS = [int(t * MAX_TEST_SIZE / len(THRESHOLDS)) for t in range(len(THRESHOLDS))]
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        decision_dict = {'DECISION': 'continue', 'ESTIMATED_DIFFERENCE': None}
        N = A_SUCCESS + A_FAIL + B_SUCCESS + B_FAIL
        AT_CHECKPOINT = N in CHECKPOINTS 
        LAST_SAMPLE = N >= MAX_TEST_SIZE
        if LAST_SAMPLE:
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
            if Z > 0:
                decision_dict['DECISION'] = 'B'
            else:
                decision_dict['DECISION'] = 'A'
        elif AT_CHECKPOINT:
            for i in range(len(CHECKPOINTS)):
                if N == CHECKPOINTS[i]:
                    break
            T = THRESHOLDS[i]
            TMP = get_Z(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL)
            Z = TMP[0]
            PAST_THRESHOLD = abs(Z) > T
            if PAST_THRESHOLD:
                decision_dict['ESTIMATED_DIFFERENCE'] = TMP[1]
                if Z > 0:
                    decision_dict['DECISION'] = 'B'
                else:
                    decision_dict['DECISION'] = 'A'
        return decision_dict
    return decision_function

DATA_FILE = 'test_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)
evaluate_all(data_dict, obrien_fleming)