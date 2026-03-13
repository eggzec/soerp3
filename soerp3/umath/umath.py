"""
Generalizes mathematical operators that work on numeric objects (from the math
module) compatible with objects with uncertainty distributions
"""

import numpy as np

from soerp3 import _make_UF_compatible_object


# import sys

__author__ = "Abraham Lee"

__all__ = []

e = np.e
pi = np.pi

__all__.append("e")
__all__.append("pi")


def abs(x):
    return _make_UF_compatible_object(np.abs(x))


__all__.append("abs")


def acos(x):
    return _make_UF_compatible_object(np.arccos(x))


__all__.append("acos")


def acosh(x):
    return _make_UF_compatible_object(np.arccosh(x))


__all__.append("acosh")


def acot(x):
    return _make_UF_compatible_object(np.arctan(1 / x))


__all__.append("acot")


def acoth(x):
    return _make_UF_compatible_object(np.arctanh(1 / x))


__all__.append("acoth")


def acsc(x):
    return _make_UF_compatible_object(np.arcsin(1 / x))


__all__.append("acsc")


def acsch(x):
    return _make_UF_compatible_object(np.arcsinh(1 / x))


__all__.append("acsch")


def asec(x):
    return _make_UF_compatible_object(np.arccos(1 / x))


__all__.append("asec")


def asech(x):
    return _make_UF_compatible_object(np.arccosh(1 / x))


__all__.append("asech")


def asin(x):
    return _make_UF_compatible_object(np.arcsin(x))


__all__.append("asin")


def asinh(x):
    return _make_UF_compatible_object(np.arcsinh(x))


__all__.append("asinh")


def atan(x):
    return _make_UF_compatible_object(np.arctan(x))


__all__.append("atan")


def atan2(y, x):
    return _make_UF_compatible_object(np.arctan2(y, x))


__all__.append("atan2")


def atanh(x):
    return _make_UF_compatible_object(np.arctanh(x))


__all__.append("atanh")


def ceil(x):
    return _make_UF_compatible_object(np.ceil(x))


__all__.append("ceil")


def cos(x):
    return _make_UF_compatible_object(np.cos(x))


__all__.append("cos")


def cosh(x):
    return _make_UF_compatible_object(np.cosh(x))


__all__.append("cosh")


def cot(x):
    return _make_UF_compatible_object(1 / np.tan(x))


__all__.append("cot")


def coth(x):
    return _make_UF_compatible_object(1 / np.tanh(x))


__all__.append("coth")


def csc(x):
    return _make_UF_compatible_object(1 / np.sin(x))


__all__.append("csc")


def csch(x):
    return _make_UF_compatible_object(1 / np.sinh(x))


__all__.append("csch")


def degrees(x):
    return _make_UF_compatible_object(np.degrees(x))


__all__.append("degrees")


def erf(x):
    return _make_UF_compatible_object(np.math.erf(x))


__all__.append("erf")


def erfc(x):
    return _make_UF_compatible_object(np.math.erfc(x))


__all__.append("erfc")


def exp(x):
    return _make_UF_compatible_object(np.exp(x))


__all__.append("exp")


def expm1(x):
    return _make_UF_compatible_object(np.expm1(x))


__all__.append("expm1")


def fabs(x):
    return _make_UF_compatible_object(np.fabs(x))


__all__.append("fabs")


def factorial(x):
    return _make_UF_compatible_object(np.math.factorial(x))


__all__.append("factorial")


def floor(x):
    return _make_UF_compatible_object(np.floor(x))


__all__.append("floor")


def gamma(x):
    return _make_UF_compatible_object(np.math.gamma(x))


__all__.append("gamma")


def lgamma(x):
    return _make_UF_compatible_object(np.math.lgamma(x))


__all__.append("lgamma")


def hypot(x, y):
    return _make_UF_compatible_object(np.hypot(x, y))


__all__.append("hypot")


def ln(x):
    return _make_UF_compatible_object(np.log(x))


__all__.append("ln")


def log(x, base):
    return _make_UF_compatible_object(np.log(x) / np.log(base))


__all__.append("log")


def log10(x):
    return _make_UF_compatible_object(np.log10(x))


__all__.append("log10")


def log1p(x):
    return _make_UF_compatible_object(np.log1p(x))


__all__.append("log1p")


def pow(x):
    return _make_UF_compatible_object(np.power(x, 2))


__all__.append("pow")


def radians(x):
    return _make_UF_compatible_object(np.radians(x))


__all__.append("radians")


def sec(x):
    return _make_UF_compatible_object(1 / np.cos(x))


__all__.append("sec")


def sech(x):
    return _make_UF_compatible_object(1 / np.cosh(x))


__all__.append("sech")


def sin(x):
    return _make_UF_compatible_object(np.sin(x))


__all__.append("sin")


def sinh(x):
    return _make_UF_compatible_object(np.sinh(x))


__all__.append("sinh")


def sqrt(x):
    return _make_UF_compatible_object(np.sqrt(x))


__all__.append("sqrt")


def tan(x):
    return _make_UF_compatible_object(np.tan(x))


__all__.append("tan")


def tanh(x):
    return _make_UF_compatible_object(np.tanh(x))


__all__.append("tanh")


def trunc(x):
    return _make_UF_compatible_object(np.trunc(x))


__all__.append("trunc")
