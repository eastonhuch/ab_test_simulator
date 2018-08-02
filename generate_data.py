import random
import pickle
import os
import numpy as np

def generate_data(ALPHA_A, BETA_A, ALPHA_B, BETA_B, MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, PROPORTION_IN_B=0.5):
    data = np.zeros((NUM_TESTS, MAX_TEST_SIZE, 6))
    p_differences = np.zeros(NUM_TESTS)
    for i in range(NUM_TESTS):
        if i % 10 == 0:
            print(str(i) + '/' + str(NUM_TESTS))
        curr = [0] * 6 # [success A, fail A, loss A, success B, fail B, loss B]
        p_A = np.random.beta(ALPHA_A, BETA_A, 1)
        p_B = np.random.beta(ALPHA_B, BETA_B, 1)
        p_differences[i] = float(p_B - p_A)
        abs_diff = abs(p_differences[i])
        
        n_remaining = HORIZON_LENGTH
        if p_B > p_A:
            regret_indexes = (0, 1)
            loss_correct_idx = 5
            loss_incorrect_idx = 2
        else:
            regret_indexes = (3, 4)
            loss_correct_idx = 2
            loss_incorrect_idx = 5

        for j in range(MAX_TEST_SIZE):
            n_remaining -= 1
            if random.random() > PROPORTION_IN_B:
                if random.random() < p_A:
                    curr[0] += 1
                else:
                    curr[1] += 1
            else:
                if random.random() < p_B:
                    curr[3] += 1
                else:
                    curr[4] += 1

            curr[loss_correct_idx] = abs_diff * np.sum([curr[k] for k in regret_indexes])
            curr[loss_incorrect_idx] = curr[loss_correct_idx] + abs_diff * n_remaining
            data[i][j] = curr
            
    return {'DATA':data, 'P_DIFFERENCES':p_differences}

def get_mme(MEAN, STDDEV):
    M = float(MEAN)
    S = float(STDDEV)
    a = M * (M * (1 - M) / (S ** 2) - 1)
    b = a * (1 - M) / M
    return (float(a), float(b))

def save_data(PRIORS, MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, PATH='./data.pkl', PROPORTION_IN_B=0.5):
    out_dict = {'A_PRIORS':PRIORS[0],
                'MAX_TEST_SIZE':MAX_TEST_SIZE,
                'HORIZON_LENGTH':HORIZON_LENGTH,
                'NUM_TESTS':NUM_TESTS}
    data_dict = generate_data(PRIORS[0][0], PRIORS[0][1], PRIORS[1][0], PRIORS[1][1],
                              MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, PROPORTION_IN_B)
    out_dict['DATA'] = data_dict['DATA']
    out_dict['P_DIFFERENCES'] = data_dict['P_DIFFERENCES']

    # Maximum number of bytes pickle can write without a problem
    # See https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    # Even this solution gives me issues with large files
    MAX_BYTES = 2**31-1
    data_out = pickle.dumps(out_dict)
    with open(PATH, 'wb') as f:
        for idx in range(0, len(data_out), MAX_BYTES):
            f.write(data_out[idx:idx+MAX_BYTES])
        print('Saved data to ' + PATH)
    return

def read_data(PATH):
    MAX_BYTES = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(PATH)
    with open(PATH, 'rb') as f_in:
        for _ in range(0, input_size, MAX_BYTES):
            bytes_in += f_in.read(MAX_BYTES)
    DATA = pickle.loads(bytes_in)
    return DATA