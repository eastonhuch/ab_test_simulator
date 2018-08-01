import numpy as np

def evaluate_one_test(decision_function, DATA, MAX_TEST_SIZE):
    # Decision functions return a dictionary with these keys
    decision_dict = {'DECISION':'continue', 'ESTIMATED_DIFFERENCE':None}
    
    for i in range(MAX_TEST_SIZE):
        decision_dict = decision_function(DATA[i, 0], DATA[i, 1], DATA[i, 3], DATA[i, 4])
        if decision_dict['DECISION'] != 'continue':
            break
    
    if decision_dict['DECISION'] == 'continue':
        print('Your decision function failed to decide within the alotted test size')
        print('Defaulting to A arm')
        decision_dict['DECISION'] = 'A'
    if decision_dict['DECISION'] == 'A':
        loss_idx = 2
    elif decision_dict['DECISION'] == 'B':
        loss_idx = 5
    else:
        print('Your decision function returned an invalid decision: ' + decision_dict['DECISION'])
        print('Options are "A", "B", and "continue"')
        raise ValueError
        
    results = {'LOSS':DATA[i, loss_idx],
               'NUM_SAMPLES':i+1,
               'ESTIMATED_DIFFERENCE':decision_dict['ESTIMATED_DIFFERENCE'],
               'CHOSE_B': decision_dict['DECISION'] == 'B'}
    return results

def evaluate_all(D, create_decision_function):
    decision_function = create_decision_function(D['A_PRIORS'][0], D['A_PRIORS'][1], 
                                                 D['HORIZON_LENGTH'], D['MAX_TEST_SIZE'])
    NUM_TESTS = len(D['DATA'])
    results= []
    print('Validating rule on sample data')
    for i, t in enumerate(D['DATA']):
        if i % 10 == 0:
            print(str(i) + '/' + str(NUM_TESTS))
        results.append(evaluate_one_test(decision_function, t, D['MAX_TEST_SIZE']))

    
    AVG_LOSS   = np.mean([r['LOSS'] for r in results])
    AVG_LENGTH = np.mean([r['NUM_SAMPLES'] for r in results])
    ESTIMATED_DIFFERENCES = np.array([r['ESTIMATED_DIFFERENCE'] for r in results])
    CHOSE_B = np.array([r['CHOSE_B'] for r in results])
    WIN  = D['P_DIFFERENCES'] > 0
    DEVIATIONS = ESTIMATED_DIFFERENCES - D['P_DIFFERENCES']
    DEVIATIONS_B = DEVIATIONS[CHOSE_B]
    
    if len(DEVIATIONS_B) > 0:   
        BIAS = np.mean(DEVIATIONS_B)
        MSE  = np.mean(DEVIATIONS_B**2)
    else:
        print('WARNING: Your rule never chose the B arm')
        BIAS = float('NaN')
        MSE  = float('NaN')
        
    TP = np.sum(WIN & CHOSE_B)
    FP = np.sum(~WIN & CHOSE_B)
    TN = np.sum(~WIN & ~CHOSE_B)
    FN = np.sum(WIN & ~CHOSE_B)
    
    result_dict = {'Average loss': AVG_LOSS,
                   'Average number of samples': AVG_LENGTH,
                   'Estimate bias': BIAS, 'Estimate MSE': MSE,
                   'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return result_dict