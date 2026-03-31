"""
Created on Tue Apr  9 15:48:17 2013

Overview
--------
The ``soerp`` package is the python equivalent of N. D. Cox's original SOERP
code written in Fortran. See the documentation in UncertainVariable for more
details and a reference to his work.

Credits
-------
A lot of code here was inspired/evolved from the `uncertainties`_ package by
`Eric O. LEBIGOT`_. I'm grateful to him for his support and good work.

.. _uncertainties: http://pypi.python.org/pypi/uncertainties
.. _Eric O. LEBIGOT: http://www.linkedin.com/pub/eric-lebigot/22/293/277

"""

from importlib.metadata import PackageNotFoundError, version

from .distributions import Beta, Chi2, Exp, F, Gamma, LogN, N, T, Tri, U, Weib
from .method_of_moments import raw2central
from .statistics import correlation_matrix, covariance_matrix
from .uncertain_function import (
    CONSTANT_TYPES,
    UncertainFunction,
    _combine_op,
    _make_UF_compatible_object,
    _unary_op,
    to_uncertain_func,
)
from .uncertain_variable import UncertainVariable, uv


__author__ = "Abraham Lee"

try:  # noqa: RUF067, RUF100
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CONSTANT_TYPES",
    "Beta",
    "Chi2",
    "Exp",
    "F",
    "Gamma",
    "LogN",
    "N",
    "T",
    "Tri",
    "U",
    "UncertainFunction",
    "UncertainVariable",
    "Weib",
    "_combine_op",
    "_make_UF_compatible_object",
    "_unary_op",
    "correlation_matrix",
    "covariance_matrix",
    "raw2central",
    "to_uncertain_func",
    "uv",
]
