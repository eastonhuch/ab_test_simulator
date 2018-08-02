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
from bayes_95 import bayes_95
from current_approach import current_approach

function_list = [evan_miller, evan_miller_most_likely, 
                expected_conversion_loss, most_likely, naive_peeking, 
                obrien_fleming, bayes_95, current_approach]

results = []
DATA_FILE = 'small_data.pkl'
RESULTS_FILE = 'results.pkl'
data_dict = read_data(DATA_FILE)

for i, f in enumerate(function_list):
    print('Evaluting decision function ' + str(i))    
    results.append(evaluate_all(data_dict, f))
    # Checkpointing in case something breaks
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
#with open(RESULTS_FILE, 'rb') as input:
#    results = pickle.load(input)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('results.csv')

