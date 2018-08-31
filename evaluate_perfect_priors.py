import pickle
import pandas as pd
from generate_data import read_data
from evaluation_functions import evaluate_all

# These are the functions we will be testing
from perfect_priors import loss_diff, loss_ratio, bayes_p

BAYES_PROBS = [0.90, 0.95, 0.99]
BAYES_P_LIST = []
for p in BAYES_PROBS:
    #b = bayes_p(p)
    #b.__name__ = 'bayes_' + str(int(p * 100))
    #BAYES_P_LIST.append(b)
    b = bayes_p(p, True)
    b.__name__ = 'bayes_' + str(int(p * 100)) + '_hard'
    BAYES_P_LIST.append(b)

#FUNCTION_LIST = [loss_diff, loss_ratio] + BAYES_P_LIST
FUNCTION_LIST = BAYES_P_LIST

results = []
DATA_FILE = 'test_data.pkl'
data_dict = read_data(DATA_FILE)

for i, f in enumerate(FUNCTION_LIST):
    print('Evaluting decision function ' + str(i + 1) + ': ' + f.__name__)
    d = evaluate_all(data_dict, f)
    d['Rule name'] = f.__name__
    results.append(d)
    
COLS = ['Rule name', 'Average loss', 'Average number of samples',
        'Estimate bias', 'Estimate MSE', 'TP', 'TN', 'FP', 'FN']
results_df = pd.DataFrame(results).sort_values('Average loss').reset_index(drop=True).loc[:, COLS]
results_df.to_csv('results_perfect_priors.csv', index=False)
print(results_df)