# API Reference

soerp3 provides a Python implementation of the SOERP method (Cox 1979) for second-order error propagation. See the [Theory](theory.md) and [Quickstart](quickstart.md) for mathematical background and usage.

## Main Features

- Transparent calculations with automatic derivatives
- Basic NumPy support
- Nearly all standard math module functions supported via `soerp3.umath` (e.g., `sin`, `exp`, `sqrt`, etc.)
- Analytical derivatives up to second order
- Easy continuous distribution constructors:
	- `N(mu, sigma)`: Normal
	- `U(a, b)`: Uniform
	- `Exp(lamda, [mu])`: Exponential
	- `Gamma(k, theta)`: Gamma
	- `Beta(alpha, beta, [a, b])`: Beta
	- `LogN(mu, sigma)`: Log-normal
	- `Chi2(k)`: Chi-squared
	- `F(d1, d2)`: F-distribution
	- `Tri(a, b, c)`: Triangular
	- `T(v)`: T-distribution
	- `Weib(lamda, k)`: Weibull

## Core Classes and Functions

- `uv`: Uncertain variable constructor (accepts moments or a scipy.stats distribution)
- `N`, `U`, `Exp`, `Gamma`, `Chi2`, ...: Distribution shortcuts for common continuous distributions
- `umath`: Math functions for uncertain variables
- `describe()`: Print mean, variance, skewness, kurtosis
- `moments()`: Return moments of a variable
- `d()`, `d2()`, `d2c()`: First and second derivatives, mixed derivatives
- `gradient()`, `hessian()`: Vector/matrix of derivatives
- `error_components(pprint=True/False)`: Variance decomposition and error component breakdown

## Example Workflows

soerp3 supports both direct moment input and distribution-based construction. You can:

- Create uncertain variables from moments, scipy.stats distributions, or constructors
- Combine variables using arithmetic and math functions
- Compute and print all moments, derivatives, and error components

See the [Quickstart](quickstart.md) for full code examples, including:

- Assembly stack-up
- Orifice flow
- Manufacturing tolerance stackup
- Scheduling facilities
- Two-bar truss

All examples demonstrate both moment-based and distribution-based usage, as well as advanced features like error decomposition.
