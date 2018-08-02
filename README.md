# AB Test Simulator

## Background
This project simulates AB tests with 2 arms and a binary outcome.
The user can specify his or her own testing rule and validate it against simulated data.

## Parameter Definitions
* `ALPHA` and `BETA`: The parameters for the beta distribution (see https://en.wikipedia.org/wiki/Beta_distribution)
from which the conversion rate on the A-arm is drawn.
Your test function will know these parameters for the A-arm but not the B-arm.
The mean of the B-arm's distribution is the same as the A-arm's, but the standard deviation is unknown.
If your method requires specifying a prior distribution for the B-arm,
you must use this information to choose reasonable parameters.

* `MAX_TEST_SIZE`: The maximum number of users that can be in the test.

* `HORIZON_LENGTH`: The number of users that will be used to calculate loss.
The horizon length includes users that are in the test.

* `NUM_TESTS`: This is the number of tests that are simulated.

* `BASELINE_CONVERSION_RATE`: This is the mean of the (beta) distribution for conversion rates on both arms.

* `A_STDDEV` and `B_STDDEV`: The standard deviation of the conversion rate distribution on the A- and B-arms respectively.
Together with the `BASELINE_CONVERSION_RATE`, you can use these to backsolve
for `ALPHA` and `BETA` with the `get_mme` function in `generate_data.py`.

* `A_SUCCESS`, `A_FAIL`, `B_SUCCESS`, and `B_FAIL`: The cumulative number of successes (conversions) or failures on the given arm.
For example, `A_SUCCESS=10` means 
"at this point in the test, there have been 10 successes on the A-arm."

## Usage
Open the file `example_decision_function.py` and replace the function `always_B` with your own decision function,
taking care to replace the reference to `always_B` in the last line with your own function.
Then run the whole file to see how your rule performs on some simulated data.

Notice that test rules are functions that return functions 
(i.e., the inner function--the function that is actually applied to the data--is a closure).
We adopted this design because some test rules will need to know general setup parameters, 
but we didn't want to pass them to the function every time.

Your function must return a dictionary with keys `DECISION` and `ESTIMATED_DIFFERENCE`.
Allowed values for `DECISION` are `A`, `B`, or `continue`.
Make sure to set `DECISION` to either `A` or `B` on the last iteration;
otherwise, the simulator will display a warning and default to `A`.
`ESTIMATED_DIFFERENCE` is your method's estimate of the difference between conversion rates (p<sub>B</sub> - p<sub>A</sub>).

You're free to use the functions in the provided files.
Of particular interest is the `get_mme` function, 
a function that backsolves for `ALPHA` and `BETA` given a `MEAN` and a `STDDEV`.
You may also want to peruse the helper functions used by other test rules.
For example, `current_approach.py` contains the functions `power_calculation` and `get_Z`.

When you're finished with your function, put it in it's own file and create a pull request.
Alternatively, you could just send your file to Easton, and he will add it for you.

## Issues
This project is still under development, so please report any bugs (or other shortcomings)
that you find in the course of developing test rules.
