# Theory

## Tolerance Analysis by Computer (Cox 1979)

### Abstract
An improved error propagation technique was needed for the analysis of nuclear reactor safety systems. In response, a computer program for calculating the first four moments of a second-order function was prepared and validated. This program, called SOERP, can accommodate up to 30 statistically independent random variables. Thus, the moments for any function that can be expanded in a multivariable Taylor series up to the second order can be estimated. These moments can then be used to determine a probability density function describing the dependent variable. The above two steps represent a method for evaluating a complex system's performance fluctuations that arise from random variability in the behavior of system components. Applications include tolerance limits on the performance of electrical circuits, transit times in repair facilities, and profitability calculations subject to uncertainty.

### Introduction
Engineers often need to predict the performance of equipment or processes before construction or use. This is known as *tolerance analysis* when examining the effect of part tolerances, or *uncertainty analysis* when uncertainties arise from incomplete knowledge or environmental measurements. The goal is to estimate the output scatter from random variables and use the resulting probability density function (pdf) to predict system variability. Standard practice is to determine the first four moments of the output and select an appropriate distribution.

The analysis starts with a multivariable Taylor series expansion of the system output. Truncating after first-order terms yields linear error propagation; retaining second-order terms yields more accurate, but more complex, second-order error propagation equations.

SOERP (Second-Order Error Propagation) was developed to perform rigorous second-order error propagation for functions expandable in a Taylor series with statistically independent input variables.

### Mathematical Formulation
The system output variable is expressed as:

$$
Y = Z(X_1, X_2, \ldots, X_n) - Z(\nu_{11}, \nu_{21}, \ldots, \nu_{n1})
$$

where $Z$ is approximated by a second-order polynomial:

$$
Z = b_0 + \sum_{i=1}^n b_i (X_i - \nu_{i1}) + \sum_{i=1}^n b_{ii} (X_i - \nu_{i1})^2 + \sum_{i<j} b_{ij}(X_i - \nu_{i1})(X_j - \nu_{j1})
$$

with coefficients:

$$
\begin{align}
b_0 &= Z(\nu_{11}, \nu_{21}, \ldots, \nu_{n1}) \\
b_i &= \left.\frac{\partial h}{\partial X_i}\right|_{\nu} \\
b_{ii} &= \frac{1}{2}\left.\frac{\partial^2 h}{\partial X_i^2}\right|_{\nu} \\
b_{ij} &= \left.\frac{\partial^2 h}{\partial X_i \partial X_j}\right|_{\nu}
\end{align}
$$

The second, third, and fourth central moments provide measures of variance, skewness, and kurtosis, respectively. Dimensionless skewness and kurtosis coefficients are:

$$
\beta_1 = \frac{\mu_3^2}{\mu_2^3}, \qquad \beta_2 = \frac{\mu_4}{\mu_2^2}
$$

For a normal distribution, $\beta_1 = 0$ and $\beta_2 = 3$.

### Transformation of Variables
A transformed variable is often used:

$$
W_i = \frac{X_i - \nu_{i1}}{\sigma_i}
$$

so that each transformed variable has zero mean and unit variance. This simplifies specification of higher-order central moments, which then depend only on the form of the probability distribution.

### Examples
#### Assembly Example
An assembly is built from three components with output:

$$
Z = \frac{X_1 X_2}{15(1.5 + X_3)}
$$

Nominal values: $\nu_{11}=24.0$, $\nu_{21}=37.0$, $\nu_{31}=0.5$. The first two variables are normally distributed, the third is exponentially distributed. A second-order Taylor expansion is performed, coefficients are computed, and the resulting polynomial is transformed into standardized variables. The output is non-Gaussian with significant skewness and kurtosis, best represented by a Pearson Type IV distribution.

#### Orifice Flow Example
The volumetric gas flow measured by an orifice meter is:

$$
Q = \frac{C}{\sqrt{M}} \frac{520 H P}{t + 460}
$$

Assuming normally distributed inputs, second-order error propagation yields a nearly Gaussian distribution for $Q$, with variance slightly lower than that obtained via linear propagation. Variance decomposition shows that pressure differential contributes the most to uncertainty in flow rate.

### Availability of SOERP
The SOERP program, written in FORTRAN IV for IBM 360/75 and CDC Cyber 173/76 computers, is available from the Argonne Code Center:

> Mrs. Margaret Butler  
> Argonne Code Center  
> Building 203--Room C230  
> 9700 South Cass Avenue  
> Argonne, Illinois 60439

### Acknowledgments
The author thanks C. F. Miller for programming and validation work with SOERP and Dr. Samuel S. Shapiro for valuable suggestions.

### References
- A. H. Bowker and G. J. Lieberman, *Engineering Statistics*, Prentice--Hall, 1959.
- R. S. Burington and D. C. May, *Handbook of Probability and Statistics*, McGraw--Hill, 1970.
- N. D. Cox, "Comparison of Two Uncertainty Analysis Methods," *Nuclear Science and Engineering*, vol. 64, no. 1, 1977.
- G. J. Hahn and S. S. Shapiro, *Statistical Models in Engineering*, Wiley, 1967.
- N. L. Johnson and S. Kotz, *Continuous Univariate Distributions*, Houghton Mifflin, 1970.
- H. H. Ku, "Notes on the Use of Propagation of Error Formulas," NBS Special Publication 300, 1969.
- W. Volk, *Applied Statistics for Engineers*, McGraw--Hill, 1969.
