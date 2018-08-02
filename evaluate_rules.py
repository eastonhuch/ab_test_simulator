import pickle
import pandas as pd
from generate_data import read_data
from evaluation_functions import evaluate_all

# These are the functions we will be testing
from evan_miller import evan_miller, evan_miller_most_likely
from expected_conversion_loss import expected_conversion_loss
from most_likely import most_likely
from naive_peeking import naive_peeking
from obrien_fleming import obrien_fleming
from bayes_p import bayes_p
from current_approach import current_approach

BAYES_PROBS = [0.90, 0.95, 0.99]
BAYES_P_LIST = []
for p in BAYES_PROBS:
    b = bayes_p(p)
    b.__name__ = 'bayes_' + str(int(p * 100))
    BAYES_P_LIST.append(b)

FUNCTION_LIST = [evan_miller, evan_miller_most_likely, 
                expected_conversion_loss, most_likely, naive_peeking, 
                current_approach, obrien_fleming] + BAYES_P_LIST

results = []
DATA_FILE = 'test_data.pkl'
RESULTS_FILE = 'results.pkl'
data_dict = read_data(DATA_FILE)

for i, f in enumerate(FUNCTION_LIST):
    print('Evaluting decision function ' + str(i + 1))
    d = evaluate_all(data_dict, f)
    d['Rule name'] = f.__name__
    results.append(d)
    # Checkpointing in case something breaks
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
#with open(RESULTS_FILE, 'rb') as input:
#    results = pickle.load(input)

COLS = ['Rule name', 'Average loss', 'Average number of samples',
        'Estimate bias', 'Estimate MSE', 'TP', 'TN', 'FP', 'FN']
results_df = pd.DataFrame(results).sort_values('Average loss').reset_index(drop=True).loc[:, COLS]
results_df.to_csv('results.csv', index=False)
print(results_df)