# AB Test Simulator

## Background
This project simulates AB tests with 2 arms and a binary outcome.
The user can specify different baseline conversion rates,
the variation around these rates (standard deviation),
the maximum test length,
and the horizon over which to optimize.

## Parameter Definitions
* `ALPHA` and `BETA`: The conversion rates on both arms are drawn from beta distributions (https://en.wikipedia.org/wiki/Beta_distribution). 
Your test function will know that parameters for the A arm but not the B arm.
The mean of the B-arm's distribution is the same as the A-arm's, but the standard deviation is unknown.
If your method requires specifying a prior distribution for the B-arm, you will have to choose reasonable values with this information.

* `MAX_TEST_SIZE`: The maximum number of users that can be in the test.

* `HORIZON_LENGTH`: The number of users that will be used to calculate loss.
The horizon length includes users that are in the test.

* `NUM_TESTS`: This is the number of tests that are simulated.

* `BASELINE_CONVERSION_RATE`: This is the mean of the (beta) distribution for conversion rates on both arms.

* `A_STDDEV` and `B_STDDEV`: The standard deviation of the conversion rate distribution on the A- and B-arms respectively.
Together with the `BASELINE_CONVERSION_RATE`, you can use these to backsolve for the `ALPHA` and `BETA` using the `get_mme` function in `generate_data.py`.

## Usage
Open the file `example_decision_function.py` and replace the function `always_B` with your own decision function.
Notice that test rules are functions that return functions.
We adopted this design because some test rules will need to know general setup parameters, but we didn't want to pass them to the function every time.

Your function must return a dictionary with keys `DECISION` and `ESTIMATED_DIFFERENCE`.
Allowed values for `DECISION` are `A`, `B`, or `continue`.

You're free to use the functions in the provided files.
The that are likely to be most helpful are `get_mme` (a function that backsolves for method of moments estimators) and `generate_data`, both of which can be found in `generate_data.py`.

## Issues
This project is still under development, so please report any bugs (or other shortcomings) that you find in the course of developing test rules.
