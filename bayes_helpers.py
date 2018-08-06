import numpy as np

def sample_posterior(ALPHA_A, BETA_A, ALPHA_B, BETA_B, 
                     N_A, N_B, N_REMAINING, N_SAMPLES=500):
    A = np.random.beta(ALPHA_A, BETA_A, N_SAMPLES)
    B = np.random.beta(ALPHA_B, BETA_B, N_SAMPLES)
    DIFF = B - A
    B_BETTER_IDX = DIFF > 0
    A_BETTER_IDX = ~B_BETTER_IDX

    # Expected loss if you choose A
    loss_A_better = -(DIFF * A_BETTER_IDX * N_B)
    loss_B_better = DIFF * B_BETTER_IDX * (N_A + N_REMAINING)
    LOSS_A = np.mean(loss_A_better + loss_B_better)

    # Expected loss if you choose B
    loss_A_better = -(DIFF * A_BETTER_IDX * (N_B + N_REMAINING))
    loss_B_better = DIFF * B_BETTER_IDX * N_A
    LOSS_B = np.mean(loss_A_better + loss_B_better)
    
    result = {'LOSS_A': LOSS_A, 'LOSS_B': LOSS_B, 'P_DIFF': np.mean(DIFF), 'P_WIN':np.mean(DIFF > 0)}
    return result

def generate_training_data(ALPHA_A, BETA_A, ALPHA_B, BETA_B,
                           MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS,
                           N_SAMPLES=500, PROPORTION_IN_B=0.5, JUMP=100, RATIO=False):
    print('Generating training data')
    ROWS_PER_TEST = int(MAX_TEST_SIZE / JUMP) # Make sure they're evenly divisible
    data = np.zeros((NUM_TESTS, ROWS_PER_TEST, 7))
    for i in range(NUM_TESTS):
        if i % 10 == 0:
            print(str(i) + '/' + str(NUM_TESTS))
        # [success A, fail A, loss A, success B, fail B, loss B, diff in expected loss]
        curr = [0] * 7
        p_A = np.random.beta(ALPHA_A, BETA_A, 1)
        p_B = np.random.beta(ALPHA_B, BETA_B, 1)
        p_diff = abs(p_B - p_A)
        n_remaining = HORIZON_LENGTH
        if p_B > p_A:
            regret_idexes = (0, 1)
            loss_correct_idx = 5
            loss_incorrect_idx = 2
        else:
            regret_idexes = (3, 4)
            loss_correct_idx = 2
            loss_incorrect_idx = 5

        for j in range(ROWS_PER_TEST):
            n_remaining -= JUMP
            n_in_B = np.random.binomial(JUMP, PROPORTION_IN_B)
            n_in_A = JUMP - n_in_B
            a_success = np.random.binomial(n_in_A, p_A)
            b_success = np.random.binomial(n_in_B, p_B)
            curr[0] += a_success
            curr[1] += n_in_A - a_success
            curr[3] += b_success
            curr[4] += n_in_B - b_success
            curr[loss_correct_idx] = p_diff * np.sum([curr[k] for k in regret_idexes])
            curr[loss_incorrect_idx] = curr[loss_correct_idx] + p_diff * n_remaining
            posterior = sample_posterior(ALPHA_A + curr[0], BETA_A + curr[1],
                                         ALPHA_B + curr[3], BETA_B + curr[4],
                                         curr[0] + curr[1], curr[3] + curr[4],
                                         n_remaining, N_SAMPLES)
            curr[6] = posterior['LOSS_B'] / posterior['LOSS_A'] if RATIO else posterior['LOSS_B'] - posterior['LOSS_A']
            data[i][j] = curr
    print('Finished')
    return data

def get_threshold(ALPHA_A, BETA_A, ALPHA_B, BETA_B, MAX_TEST_SIZE, HORIZON_LENGTH,
                  NUM_TESTS, MIN_THRESHOLD, MAX_THRESHOLD, NUM_THRESHOLDS, N_SAMPLES=500,
                  PROPORTION_IN_B=0.5, JUMP=100, RATIO=False):
    
    DATA = generate_training_data(ALPHA_A, BETA_A, ALPHA_B, BETA_B,
                                  MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS,
                                  N_SAMPLES, PROPORTION_IN_B, JUMP, RATIO)
    THRESHOLDS = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, NUM_THRESHOLDS)

    average_loss = []
    ROWS_PER_TEST = int(MAX_TEST_SIZE / JUMP)
    for j, t in enumerate(THRESHOLDS):
        print('Testing threshold ' + str(j + 1) + ' of ' + str(NUM_THRESHOLDS))
        total_loss = 0
        for d in DATA:
            for i, r in enumerate(d):
                if RATIO:
                    if r[6] > t or r[6] < 1/t:
                        loss = r[2] if r[6] > 1 else r[5]
                        total_loss += loss
                elif abs(r[6]) > t or i >= (ROWS_PER_TEST - 1):
                    loss = r[2] if r[6] > 0 else r[5]
                    total_loss += loss
                    break
        average_loss.append(total_loss/float(len(DATA)))
        
    print('Thresholds')
    print(THRESHOLDS)
    print('Average loss for each threshold')
    print(average_loss)
    BEST_IDX = np.argmin(average_loss)
    BEST_THRESHOLD = THRESHOLDS[BEST_IDX]
    BEST_LOSS = average_loss[BEST_IDX]
    print('Best loss: ' + str(BEST_LOSS))
    print('Theshold: ' + str(BEST_THRESHOLD))

    return BEST_THRESHOLD
