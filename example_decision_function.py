from generate_data import get_mme, save_data, read_data
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
NUM_TESTS = 8000

# The conversion rates are drawn from beta distributions with these parameters
PRIORS = (get_mme(BASELINE_CONVERSION_RATE, A_STDDEV),
          get_mme(BASELINE_CONVERSION_RATE, B_STDDEV))

# Now we use those priors to generate some data to test your decision function
# This line creates your sample data
#save_data(PRIORS, MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, 'sample_data.pkl')

# This as an example decision function that always chooses the B arm
def always_B(ALPHA, BETA, HORIZON_LENGTH, MAX_TEST_SIZE):
    def decision_function(A_SUCCESS, A_FAIL, B_SUCCESS, B_FAIL):
        return {'DECISION': 'B', 'ESTIMATED_DIFFERENCE': 0.01}
    return decision_function

# Obtain average loss for your function
DATA_FILE = 'small_data.pkl'
data_dict = read_data(DATA_FILE)
evaluate_all(data_dict, always_B)