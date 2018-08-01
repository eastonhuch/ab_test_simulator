# AB Test Simulator

## Background
This project simulates AB tests with 2 arms and a binary outcome.
The user can specify different baseline conversion rates,
the variation around these rates (standard deviation),
the maximum test length,
and the horizon over which to optimize.

## Usage
Open the file `example_decision_function.py` and replace the function `always_B` with your own decision function
