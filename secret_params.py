import generate_data as gd

# Competitors know these values
MAX_TEST_SIZE = 10000
HORIZON_LENGTH = 200000
NUM_TESTS = 8000
BASELINE_CONVERSION_RATE = 0.1
A_STDDEV = 0.002
# This one is secret
B_STDDEV = 0.01

# The conversion rates are drawn from beta distributions with these parameters
PRIORS = (gd.get_mme(BASELINE_CONVERSION_RATE, A_STDDEV),
          gd.get_mme(BASELINE_CONVERSION_RATE, B_STDDEV))

gd.save_data(PRIORS, MAX_TEST_SIZE, HORIZON_LENGTH, NUM_TESTS, 'test_data.pkl')