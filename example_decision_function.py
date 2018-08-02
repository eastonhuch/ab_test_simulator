import pickle
import generate_data as gd
from evaluation_functions import evaluate_all

# These are the parameter values that will be used to validate your decision function
MAX_TEST_SIZE = 10000
HORIZON_LENGTH = 200000
BASELINE_CONVERSION_RATE = 0.1
A_STDDEV = 0.002
# This one is secret; the value below is just an example
B_STDDEV = 0.02
# This one could potentially change, but it doesn't affect your rule's performance
# A higher number of tests will give greater precision around our estimates
NUM_TESTS = 10000

# The conversion rates are drawn from beta distributions with these parameters
PRIORS = (gd.get_mme(BASELINE_CONVERSION_RATE, A_STDDEV),
          gd.get_mme(BASELINE_CONVERSION_RATE, B_STDDEV))

# Now we use those priors to generate some data to test your decision function
# NOTE: I've provided you with a data file ('sample_data.pkl')
# so you shouldn't need to run this line
gd.save_data(PRIORS, MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, 'sample_data.pkl')

# This as an example decision function that always chooses the B arm
def always_B(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        return {'DECISION': 'B', 'ESTIMATED_DIFFERENCE': 0.01}
    return decision_function

# Obtain average loss for your function
DATA_FILE = 'sample_data.pkl'
with open(DATA_FILE, 'rb') as input:
    data_dict = pickle.load(input)
evaluate_all(data_dict, always_B)