# Quickstart
This quickstart demonstrates how to use `soerp3` for second-order error propagation, following the examples in Cox (1979) and the provided soerp_examples.py. For mathematical background, see the [Theory](theory.md) section.
## Importing and Creating Variables
```python
# Normal distribution, mean=10, std=1 (using moments)
from soerp3 import uv

x = uv([10, 1, 0, 3, 0, 15, 0, 105])
print(x)
```

```python
# Normal distribution, using scipy.stats
from soerp3 import uv
import scipy.stats as ss

x = uv(rv=ss.norm(loc=10, scale=1))
print(x)
```

```python
# Normal distribution, using constructor
from soerp3 import N

x = N(10, 1)
print(x)
```

```python
from soerp3 import *  # uv, N, U, Exp, Gamma, Chi2, umath

# Normal distribution, mean=10, std=1 (using moments)
x = uv([10, 1, 0, 3, 0, 15, 0, 105])
# Normal distribution, using scipy.stats
import scipy.stats as ss

x = uv(rv=ss.norm(loc=10, scale=1))
# Normal distribution, using constructor
x = N(10, 1)
```

## Example: Three-Part Assembly
```python
# Using moments
from soerp3 import uv

x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
# Results: Mean ≈ 1176.45, Variance ≈ 99699.68, Skewness ≈ 0.71, Kurtosis ≈ 6.16
```

```python
# Using distribution constructors
from soerp3 import N, Exp, uv

x1 = N(24, 1)
x2 = N(37, 4)
x3 = Exp(2)
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
Z.describe()

# Using moments
x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
# Results: Mean ≈ 1176.45, Variance ≈ 99699.68, Skewness ≈ 0.71, Kurtosis ≈ 6.16

# Using distribution constructors
x1 = N(24, 1)
x2 = N(37, 4)
x3 = Exp(2)
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
Z.describe()
```

## Example: Orifice Flow
```python
# Using moments
from soerp3 import uv, umath

H = uv([64, 0.25, 0, 3, 0, 15, 0, 105])
M = uv([16, 0.01, 0, 3, 0, 15, 0, 105])
P = uv([361, 4, 0, 3, 0, 15, 0, 105])
t = uv([165, 0.25, 0, 3, 0, 15, 0, 105])
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
print(Q)
# Results: Mean ≈ 1331.0, Variance ≈ 58.21, Skewness ≈ 0.011, Kurtosis ≈ 3.00
```

```python
# Using constructors
from soerp3 import N, Exp, uv, umath

H = N(64, 0.5)
M = N(16, 0.1)
P = N(361, 2)
t = N(165, 0.5)
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
print(Q)
Q.describe()

# Using moments
H = uv([64, 0.25, 0, 3, 0, 15, 0, 105])
M = uv([16, 0.01, 0, 3, 0, 15, 0, 105])
P = uv([361, 4, 0, 3, 0, 15, 0, 105])
t = uv([165, 0.25, 0, 3, 0, 15, 0, 105])
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
print(Q)
# Results: Mean ≈ 1331.0, Variance ≈ 58.21, Skewness ≈ 0.011, Kurtosis ≈ 3.00

# Using constructors
H = N(64, 0.5)
M = N(16, 0.1)
P = N(361, 2)
t = N(165, 0.5)
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
print(Q)
Q.describe()
```

## Example: Manufacturing Tolerance Stackup
```python
# Using moments (gamma distributed)
from soerp3 import uv

x = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
y = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
z = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
w = x + y + z
print(w)
# Results: Mean ≈ 4.5, Variance ≈ 0.75, Skewness ≈ 0.385, Kurtosis ≈ 3.22
```

```python
# Using Gamma constructor
from soerp3 import Gamma, uv

mn = 1.5
vr = 0.25
scale = vr / mn
shape = mn**2 / vr
x = Gamma(shape, scale)
y = Gamma(shape, scale)
z = Gamma(shape, scale)
w = x + y + z
print(w)

# Using moments (gamma distributed)
x = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
y = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
z = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])
w = x + y + z
print(w)
# Results: Mean ≈ 4.5, Variance ≈ 0.75, Skewness ≈ 0.385, Kurtosis ≈ 3.22

# Using Gamma constructor
mn = 1.5
vr = 0.25
scale = vr / mn
shape = mn**2 / vr
x = Gamma(shape, scale)
y = Gamma(shape, scale)
z = Gamma(shape, scale)
w = x + y + z
print(w)
```

## Example: Scheduling Facilities (Six Stations)
```python
# Using moments
from soerp3 import uv

s1 = uv([10, 1, 0, 3, 0, 0, 0, 0])
s2 = uv([20, 2, 0, 3, 0, 0, 0, 0])
s3 = uv([1.5, 0.25, 0.67, 3.67, 0, 0, 0, 0])
s4 = uv([10, 10, 0.63, 3.6, 0, 0, 0, 0])
s5 = uv([0.2, 0.04, 2, 9, 0, 0, 0, 0])
s6 = uv([10, 20, 0.89, 4.2, 0, 0, 0, 0])
T = s1 + s2 + s3 + s4 + s5 + s6
print(T)
# Results: Mean ≈ 51.7, Variance ≈ 33.3, Skewness ≈ 0.52, Kurtosis ≈ 3.49
```

```python
# Using constructors
from soerp3 import N, Gamma, Exp, Chi2, uv

s1 = N(10, 1)
s2 = N(20, 2**0.5)
mn1 = 1.5
vr1 = 0.25
scale1 = vr1 / mn1
shape1 = mn1**2 / vr1
s3 = Gamma(shape1, scale1)
mn2 = 10
vr2 = 10
scale2 = vr2 / mn2
shape2 = mn2**2 / vr2
s4 = Gamma(shape2, scale2)
s5 = Exp(5)
s6 = Chi2(10)
T = s1 + s2 + s3 + s4 + s5 + s6
print(T)

# Using moments
s1 = uv([10, 1, 0, 3, 0, 0, 0, 0])
s2 = uv([20, 2, 0, 3, 0, 0, 0, 0])
s3 = uv([1.5, 0.25, 0.67, 3.67, 0, 0, 0, 0])
s4 = uv([10, 10, 0.63, 3.6, 0, 0, 0, 0])
s5 = uv([0.2, 0.04, 2, 9, 0, 0, 0, 0])
s6 = uv([10, 20, 0.89, 4.2, 0, 0, 0, 0])
T = s1 + s2 + s3 + s4 + s5 + s6
print(T)
# Results: Mean ≈ 51.7, Variance ≈ 33.3, Skewness ≈ 0.52, Kurtosis ≈ 3.49

# Using constructors
s1 = N(10, 1)
s2 = N(20, 2**0.5)
mn1 = 1.5
vr1 = 0.25
scale1 = vr1 / mn1
shape1 = mn1**2 / vr1
s3 = Gamma(shape1, scale1)
mn2 = 10
vr2 = 10
scale2 = vr2 / mn2
shape2 = mn2**2 / vr2
s4 = Gamma(shape2, scale2)
s5 = Exp(5)
s6 = Chi2(10)
T = s1 + s2 + s3 + s4 + s5 + s6
print(T)
```

## Example: Two-Bar Truss
```python
# Two-bar truss example
import math
from soerp3 import N, umath

H = N(30, 5 / 3.0, tag="H")
B = N(60, 0.5 / 3.0, tag="B")
d = N(3, 0.1 / 3, tag="d")
t = N(0.15, 0.01 / 3, tag="t")
E = N(30000, 1500 / 3.0, tag="E")
rho = N(0.3, 0.01 / 3.0, tag="rho")
P = N(66, 3 / 3.0, tag="P")
pi = math.pi
wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
defl = (P * ((B / 2) ** 2 + H**2) ** (1.5)) / (2 * pi * d * t * H**2 * E)
print("wght:", wght)
print("strs:", strs)
print("buck:", buck)
print("defl:", defl)
wght.error_components(pprint=True)

H = N(30, 5 / 3.0, tag="H")
B = N(60, 0.5 / 3.0, tag="B")
d = N(3, 0.1 / 3, tag="d")
t = N(0.15, 0.01 / 3, tag="t")
E = N(30000, 1500 / 3.0, tag="E")
rho = N(0.3, 0.01 / 3.0, tag="rho")
P = N(66, 3 / 3.0, tag="P")
pi = math.pi
wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
defl = (P * ((B / 2) ** 2 + H**2) ** (1.5)) / (2 * pi * d * t * H**2 * E)
print("wght:", wght)
print("strs:", strs)
print("buck:", buck)
print("defl:", defl)
wght.error_components(pprint=True)
```

## Moments, Correlations, Derivatives

You can access moments, correlations, and derivatives for any uncertain variable:

```python
# Example: moments and derivatives
from soerp3 import N, uv, Exp
import scipy.stats as ss

x1 = N(24, 1)
x2 = N(37, 4)
x3 = N(10, 1)
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z.moments())
print(x1 - x1)
square = x1**2
print(square - x1 * x1)
print(Z.d(x1))  # First derivative
print(Z.d2(x2))  # Second derivative
print(Z.d2c(x1, x3))  # Mixed second derivative
print(Z.gradient([x1, x2, x3]))
print(Z.hessian([x1, x2, x3]))
Z.error_components(pprint=True)

x1.moments()
Z.moments()
x1 - x1
square = x1**2
square - x1 * x1
Z.d(x1)  # First derivative
Z.d2(x2)  # Second derivative
Z.d2c(x1, x3)  # Mixed second derivative
Z.gradient([x1, x2, x3])
Z.hessian([x1, x2, x3])
Z.error_components(pprint=True)

x = uv(rv=ss.norm(loc=10, scale=1))
x = N(10, 1)
x2 = N(37, 4)
x3 = Exp(2)
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
Z.moments()
x1 - x1
square = x1**2
square - x1 * x1
Z.d(x1)
Z.d2(x2)
Z.d2c(x1, x3)
Z.gradient([x1, x2, x3])
Z.hessian([x1, x2, x3])
Z.error_components(pprint=True)
```
