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

import math

import numpy as np


ADF = np.ndarray
ADV = np.ndarray

import scipy.stats as ss

from .method_of_moments import (
    soerp_numeric,
    variance_components,
    variance_contrib,
)


try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_installed = False
else:
    matplotlib_installed = True

__version_info__ = (0, 9, 6)
__version__ = ".".join(list(map(str, __version_info__)))

__author__ = "Abraham Lee"

__all__ = [
    "beta",
    "chi2",
    "correlation_matrix",
    "covariance_matrix",
    "exp",
    "f",
    "gamma",
    "logn",
    "n",
    "t",
    "tri",
    "u",
    "uv",
    "weib",
]

CONSTANT_TYPES = (float, int, complex, np.number)


def to_uncertain_func(x):
    """
    Transforms x into a constant automatically differentiated UncertainFunction
    (UF), unless it already is (in which case x is returned unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on UncertainFunction objects
    (which then cannot be considered as constants).
    """

    if isinstance(x, UncertainFunction):
        return x

    # ! In Python 2.6+, numbers.Number could be used instead, here:
    if isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return UncertainFunction(x, {}, {}, {})


class UncertainFunction:
    """
    UncertainFunction objects represent the uncertainty of a result of
    calculations with uncertain variables. Nearly all basic mathematical
    operations are supported.

    This class is mostly intended for internal use.

    """

    def __init__(self, x, lc=None, qc=None, cqc=None):
        self.x = x
        self._lc = lc if lc is not None else {}
        self._qc = qc if qc is not None else {}
        self._cqc = cqc if cqc is not None else {}

    def d(self, v=None):
        if v is None:
            return self._lc
        return self._lc.get(v, 0.0)

    def d2(self, v=None):
        if v is None:
            return self._qc
        return self._qc.get(v, 0.0)

    def d2c(self, v1=None, v2=None):
        if v1 is None and v2 is None:
            return self._cqc
        return self._cqc.get((v1, v2), 0.0)

    #    def __init__(self, *args, **kwargs):
    #
    #        # UncertainFunction doesn't need a value, but it must be defined as a
    #        # place-holder for pickling
    #        self._hash = None

    def __hash__(self):
        return id(self)

    _dist = None
    _moments = None

    @property
    def mean(self) -> float:
        """
        Mean value as a result of an uncertainty calculation
        """
        mn = self.moments(0)
        return mn

    @property
    def var(self) -> float:
        """
        Variance value as a result of an uncertainty calculation
        """
        vr = self.moments(1)
        return vr

    @property
    def std(self) -> float:
        """
        Standard deviation as a result of an uncertainty calculation
        """
        return self.var ** 0.5

    def skew(self):
        r"""
        Skewness coefficient value as a result of an uncertainty calculation,
        defined as::

              _____     m3
            \/beta1 = ------
                      std**3

        where m3 is the third central moment and std is the standard deviation
        """
        sk = self.moments(2)
        return sk

    @property
    def kurt(self):
        """
        Kurtosis coefficient value as a result of an uncertainty calculation,
        defined as::

                      m4
            beta2 = ------
                    std**4

        where m4 is the fourth central moment and std is the standard deviation
        """
        kt = self.moments(3)
        return kt

    def moments(self, idx=None):
        """
        The first four standard moments of a distribution: mean, variance, and
        standardized skewness and kurtosis coefficients.
        """
        slc, sqc, scp, var_moments, f0 = self._get_inputs_for_soerp()
        m = soerp_numeric(slc, sqc, scp, var_moments, f0, silent=True)
        if idx is not None:
            assert (idx <= 3) and (idx >= 0), (
                "idx must be 0, 1, 2, or 3 since "
                + "only the first four moments can be calculated"
            )
            return m[idx]
        else:
            return m

    def _to_general_representation(self, str_func):
        m = self.moments()
        mn, vr, sk, kt = m[:4]
        return (
            f"uv({str_func(mn)}, {str_func(vr)}, {str_func(sk)}, {str_func(kt)})"
            if any([vr, sk, kt])
            else str_func(mn)
        )

    def __str__(self):
        return self._to_general_representation(str)

    def __repr__(self):
        return str(self)

    def describe(self):
        """
        Cleanly show what the distribution moments are:
            - Mean, Variance, Skewness and Kurtosis Coefficients
        """
        mn, vr, sk, kt = [self.moments(i) for i in [0, 1, 2, 3]]
        s = "SOERP Uncertain Value:\n"
        s += f" > Mean................... {mn: }\n"
        s += f" > Variance............... {vr: }\n"
        s += f" > Skewness Coefficient... {sk: }\n"
        s += f" > Kurtosis Coefficient... {kt: }\n"
        print(s)

    def _get_inputs_for_soerp(self):
        """
        This hidden method prepares the related variable moments and derivatives
        in preparation for method of moment calculations by standardizing the
        derivatives by moving the distribution to the origin and normalizing
        them with the distribution's standard deviation.
        """
        # any original variables will have a first derivative, so we can grab
        # them from that dictionary
        variables = self.d().keys()
        nvar = len(variables)

        # standardize the input derivatives
        # - slc: linear terms
        # - sqc: pure quadratic terms
        # - scp: cross quadratic terms
        slc = np.array([self.d(v) * v.std for v in variables])
        sqc = np.array([0.5 * self.d2(v) * v.var for v in variables])
        scp = np.zeros((nvar, nvar), dtype=float)
        for i, v1 in enumerate(variables):
            for j, v2 in enumerate(variables):
                if hash(v1) != hash(v2):
                    scp[int(i), int(j)] = self.d2c(v1, v2) * v1.std * v2.std
                else:
                    scp[int(i), int(j)] = 0.0

        # construct the arrays of standardized moments in since this is what
        # the method of moments calculations require
        var_moments = np.array(
            [
                [1, 0, 1] + [float(val) for val in list(v._moments[2:])]
                for v in variables
            ],
            dtype=float,
        )

        f0 = self.x  # from evaluation at input means

        inputs = (slc, sqc, scp, var_moments, f0)

        return inputs

    def error_components(self, pprint=False, as_eq_terms=False):
        """
        The parts of the second order approximation of the variance function,
        returned in three pieces if ``as_eq_terms`` = True, first-order
        components, pure-quadratic components, and cross-product components,
        otherwise the error components from the linear terms are added to the
        corresponding error components from the quadratic terms. Any
        cross-product term components are divided equally between the two
        factors of the cross-product.

        Optional
        --------
        pprint : bool, default is False,
            Pretty-print the error components, showing both the component and
            the percent contribution of the component
        as_eq_terms : bool, default is False,
            True to return the error components in the form of the equation
            terms (pure linear, pure quadratic, and cross-product), where both
            orders are available for the cross-product terms (i.e., (x, y) and
            (y, x) will be returned in the cross-product terms), otherwise in
            terms of the contributing UncertainVariables.

        Returns
        -------
        err_comp : dict
            A dictionary that maps the error components to the contributing
            UncertainVariables. If ``as_eq_terms=True``, then a tuple of three
            dictionaries is returned containing the 1) linear, 2) pure
            quadratic, and 3) cross-product term contributions).

        Example
        --------
        If we had a function of two variables ended up with the linear terms
        (``as_eq_terms`` = False here), ::

            >>> lc = {x:0.5, y:0.25}

        the quadratic terms::

            >>> qc = {x:0.2, y:0.1}

        and the cross-product term::

            >>> cp = {(x, y}:0.14}

        then the variables would be given the error components like this::

            >>> lc[x] + qc[x] + 0.5*cp[(x, y)] # first variable, x
            0.77
            >>> lc[y] + qc[y] + 0.5*cp[(x, y)] # second variable, y
            0.42

        """
        variables = self.d().keys()
        slc, sqc, scp, var_moments, f0 = self._get_inputs_for_soerp()
        vz = self.moments()

        # convert standardized moments back to central moments
        vz[2] = vz[2] * vz[1] ** 1.5
        vz[3] = vz[3] * vz[1] ** 2
        vz = [1] + vz  # the [1] is needed in the method of moment calculations
        vlc, vqc, vcp = variance_components(slc, sqc, scp, var_moments, vz)

        vc_lc = {}
        vc_qc = {}
        vc_cp = {}

        for i, v1 in enumerate(variables):
            # first-order terms
            vc_lc[v1] = vlc[i]
            # second-order terms (pure)
            vc_qc[v1] = vqc[i]
            # second-order terms (cross-product, both orientations returned)
            for j, v2 in enumerate(variables):
                if i < j:
                    vc_cp[v1, v2] = vcp[i, j]
                    vc_cp[v2, v1] = vcp[i, j]

        if not as_eq_terms:
            """
            I made an executive decision here (I don't know if it's documented
            anywhere). I decided that if I wanted to know the contributions
            separately for each variable, any cross-product contributions could
            be divided equally between the two variables. Whether or not this 
            is kosher, I'm not sure, but that's what I did. The "pure" terms 
            are also combined together by variable.
            """
            error_wrt_var = dict((v, 0.0) for v in variables)
            for i, v1 in enumerate(variables):
                if vc_lc.has_key(v1):
                    error_wrt_var[v1] += vc_lc[v1]
                    error_wrt_var[v1] += vc_qc[v1]
                for j, v2 in enumerate(variables):
                    if i < j:
                        if vc_cp.has_key((v1, v2)):
                            error_wrt_var[v1] += 0.5 * vc_cp[v1, v2]
                            error_wrt_var[v2] += 0.5 * vc_cp[v1, v2]
            if pprint:
                print("COMPOSITE VARIABLE ERROR COMPONENTS")
                for v in variables:
                    print(
                        f"{v} = {error_wrt_var[v]} or {np.abs(error_wrt_var[v] / vz[2]):%}"
                    )
                print(" ")  # one more for good measure
            else:
                return error_wrt_var
        else:
            """
            This section returns the full quadratic contributions as they 
            appear in the approximation, separated by linear terms, pure 
            quadratic terms and cross-product terms.
            """
            if pprint:
                vcont_lc, vcont_qc, vcont_cp = variance_contrib(
                    vlc, vqc, vcp, vz
                )
                print("*" * 65)
                print("LINEAR ERROR COMPONENTS:")
                for i, v1 in enumerate(variables):
                    if vc_lc.has_key(v1):
                        print(f"{v1} = {vc_lc[v1]} or {vcont_lc[i]:%}")
                    else:
                        print(f"{v1} = {0.0} or {0.0:%}")

                print("*" * 65)
                print("QUADRATIC ERROR COMPONENTS:")
                for i, v1 in enumerate(variables):
                    if vc_qc.has_key(v1):
                        print(f"{v1} = {vc_qc[v1]} or {vcont_qc[i]:%}")
                    else:
                        print(f"{v1} = {0.0} or {0.0:%}")

                print("*" * 65)
                print("CROSS-PRODUCT ERROR COMPONENTS:")
                for i, v1 in enumerate(variables):
                    for j, v2 in enumerate(variables):
                        if i < j:
                            if vc_cp.has_key((v1, v2)):
                                print(
                                    f"({v1}, {v2}) = {vc_cp[v1, v2]} or {vcont_cp[i, j]:%}"
                                )
                            elif vc_cp.has_key((v2, v1)):
                                print(
                                    f"({v2}, {v1}) = {vc_cp[v2, v1]} or {vcont_cp[j, i]:%}"
                                )
                            else:
                                print(f"({v1}, {v2}) = {0.0} or {0.0:%}")
                print(" ")  # one more for good measure
            else:
                return (vc_lc, vc_qc, vc_cp)

    def __add__(self, val):
        return _make_UF_compatible_object(self._op(val, op="add"))

    def __radd__(self, val):
        return _make_UF_compatible_object(self._op(val, op="radd"))

    def __mul__(self, val):
        return _make_UF_compatible_object(self._op(val, op="mul"))

    def __rmul__(self, val):
        return _make_UF_compatible_object(self._op(val, op="rmul"))

    def __sub__(self, val):
        return _make_UF_compatible_object(self._op(val, op="sub"))

    def __rsub__(self, val):
        return _make_UF_compatible_object(self._op(val, op="rsub"))

    def __truediv__(self, val):
        return _make_UF_compatible_object(self._op(val, op="truediv"))

    def __rtruediv__(self, val):
        return _make_UF_compatible_object(self._op(val, op="rtruediv"))

    def __pow__(self, val):
        return _make_UF_compatible_object(self._op(val, op="pow"))

    def __rpow__(self, val):
        return _make_UF_compatible_object(self._op(val, op="rpow"))

    def __neg__(self):
        return _make_UF_compatible_object(self._op(None, op="neg"))

    def __pos__(self):
        return _make_UF_compatible_object(self._op(None, op="pos"))

    def __abs__(self):
        return _make_UF_compatible_object(self._op(None, op="abs"))

    def _op(self, val, op):
        # Basic arithmetic fallback for UncertainFunction
        if op == "add":
            return self._arithmetic(val, lambda a, b: a + b)
        elif op == "radd":
            return self._arithmetic(val, lambda a, b: b + a)
        elif op == "sub":
            return self._arithmetic(val, lambda a, b: a - b)
        elif op == "rsub":
            return self._arithmetic(val, lambda a, b: b - a)
        elif op == "mul":
            return self._arithmetic(val, lambda a, b: a * b)
        elif op == "rmul":
            return self._arithmetic(val, lambda a, b: b * a)
        elif op == "truediv":
            return self._arithmetic(val, lambda a, b: a / b)
        elif op == "rtruediv":
            return self._arithmetic(val, lambda a, b: b / a)
        elif op == "pow":
            return self._arithmetic(val, lambda a, b: a**b)
        elif op == "rpow":
            return self._arithmetic(val, lambda a, b: b**a)
        elif op == "neg":
            return self._arithmetic(None, lambda a, _: -a)
        elif op == "pos":
            return self._arithmetic(None, lambda a, _: +a)
        elif op == "abs":
            return self._arithmetic(None, lambda a, _: abs(a))
        else:
            raise NotImplementedError(f"Unknown operation: {op}")

    def _arithmetic(self, val, func):
        # Propagate full uncertainty: mean, derivatives, moments
        # If val is None, unary op
        if val is None:
            x = func(self.x, None)
            lc = {v: func(d, None) for v, d in self.d().items()}
            qc = {v: func(d2, None) for v, d2 in self.d2().items()}
            cqc = {
                (v1, v2): func(d2c, None)
                for (v1, v2), d2c in self.d2c().items()
            }
        elif isinstance(val, UncertainFunction):
            x = func(self.x, val.x)
            # Linear coefficients
            lc = {}
            for v in set(self.d().keys()).union(val.d().keys()):
                lc[v] = func(self.d().get(v, 0.0), val.d().get(v, 0.0))
            # Pure quadratic coefficients
            qc = {}
            for v in set(self.d2().keys()).union(val.d2().keys()):
                qc[v] = func(self.d2().get(v, 0.0), val.d2().get(v, 0.0))
            # Cross quadratic coefficients
            cqc = {}
            all_vars = set(self.d().keys()).union(val.d().keys())
            for v1 in all_vars:
                for v2 in all_vars:
                    cqc[v1, v2] = func(
                        self.d2c().get((v1, v2), 0.0),
                        val.d2c().get((v1, v2), 0.0),
                    )
        else:
            x = func(self.x, val)
            lc = {v: func(d, val) for v, d in self.d().items()}
            qc = {v: func(d2, val) for v, d2 in self.d2().items()}
            cqc = {
                (v1, v2): func(d2c, val) for (v1, v2), d2c in self.d2c().items()
            }
        # Return new UncertainFunction with propagated uncertainty
        return UncertainFunction(x, lc, qc, cqc)

    def __eq__(self, val):
        diff = self - val
        return not (diff.mean or diff.var or diff.skew or diff.kurt)

    def __ne__(self, val):
        return not self == val

    def __lt__(self, val):
        val = to_uncertain_func(val)
        return True if float(self.mean - val.mean) < 0 else False

    def __le__(self, val):
        return (self == val) or self < val

    def __gt__(self, val):
        return not self < val

    def __ge__(self, val):
        return (self == val) or self > val

    def __nonzero__(self):
        return self != 0

    def sqrt(self):
        # Propagate sqrt for UncertainFunction
        # For scalars, just use np.sqrt(self.x)
        return _make_UF_compatible_object(
            UncertainFunction(np.sqrt(self.x), self._lc, self._qc, self._cqc)
        )


def _make_UF_compatible_object(tmp):
    if isinstance(tmp, UncertainFunction):
        return tmp
    elif (
        hasattr(tmp, "x")
        and hasattr(tmp, "d")
        and hasattr(tmp, "d2")
        and hasattr(tmp, "d2c")
    ):
        return UncertainFunction(tmp.x, tmp.d(), tmp.d2(), tmp.d2c())
    else:
        # For scalars, wrap as a constant UncertainFunction
        return UncertainFunction(tmp, {}, {}, {})


################################################################################


class UncertainVariable(UncertainFunction):
    """
    UncertainVariable objects track the effects of uncertainty, characterized
    in terms of the first four standard moments of statistical distributions
    (mean, variance, skewness and kurtosis coefficients). Most texts
    only deal with first-order models, but this class uses a full second
    order model, which requires a knowledge of the first eight central moments
    of a distribution.

    Parameters
    ----------
    moments : array-like, optional
        The first eight moments (standardized) of the uncertain variable's
        underlying statistical distribution (the first two values should be the
        mean and variance)

    rv : scipy.stats.rv_continous, optional
        If supplied, the ``moments`` kwarg is ignored and the first eight
        standardized moments are calculated internally

    tag : str, optional
        A string identifier when information about this variable is printed to
        the screen

    Notes
    -----

    For a full report on the methods behind this class, see:

        N. D. Cox, "Tolerance Analysis by Computer," Journal of Quality
        Technology, Vol. 11, No. 2, 1979.

    Here are the first eight moments of some standard distributions:

        - Normal Distribution: [0, 1, 0, 3, 0, 15, 0, 105]
        - Uniform Distribution: [0, 1, 0, 1.8, 0, 3.857, 0, 9]
        - Exponential Distribution: [0, 1, 2, 9, 44, 265, 1854, 14833]

    A distribution's raw moment (moment about the origin) is defined as::

                oo
                 /
                |
           k    |   k
        E(x ) = |  x *f(x) dx
                |
               /
               -oo

    where E(...) is the expectation operator, k is the order of the moment, and
    f(x) is the probability density function (pdf) of x.

    To convert these to central moments (moment about the mean), we can simply
    use the helper function::

        >>> moments = raw2central(raw_moments)

    or we can use the mathematical definition to calculate the kth moment as::

                     oo
                      /
                     |
                k    |        k
        E((x-mu) ) = |  (x-mu) *f(x) dx
                     |
                    /
                    -oo

    This then needs to be standardized by normalizing each of the moments
    (starting with the third moment) using the standard deviation::

        >>> sd = moment[1]**0.5
        >>> moment[k] = [moment[k]/sd**(k + 1) for k in range(2, 9)]

    The ``scipy.stats`` module contains many distributions from which we can
    easily generate these moments for any distribution. Currently, only
    ``rv_continuous`` distributions are supported. It is important to follow
    the initialization syntax for creating any kind of rv_continuous object:

        - *Location* and *Scale* values must use the kwargs ``loc`` and
          ``scale``
        - *Shape* values are passed in as arguments before the location and
          scale

    The mathematical operations that can be performed on Uncertain... objects
    will work for any moments or distribution supplied, but may not be
    misleading if the supplied moments or distribution is not accurately
    defined. Here are some guidelines for creating UncertainVariable objects
    using some of the most common statistical distributions:

    +---------------------------+-------------+-------------------+-----+---------+
    | Distribution              | scipy.stats |  args             | loc | scale   |
    |                           | class name  | (shape params)    |     |         |
    +===========================+=============+===================+=====+=========+
    | Normal(mu, sigma)         | norm        |                   | mu  | sigma   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Uniform(a, b)             | uniform     |                   | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Exponential(lamda)        | expon       |                   |     | 1/lamda |
    +---------------------------+-------------+-------------------+-----+---------+
    | Gamma(k, theta)           | gamma       | k                 |     | theta   |
    +---------------------------+-------------+-------------------+-----+---------+
    | Beta(alpha, beta, [a, b]) | beta        | alpha, beta       | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Log-Normal(mu, sigma)     | lognorm     | sigma             | mu  |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Chi-Square(k)             | chi2        | k                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | F(d1, d2)                 | f           | d1, d2            |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Triangular(a, b, c)       | triang      | c                 | a   | b-a     |
    +---------------------------+-------------+-------------------+-----+---------+
    | Student-T(v)              | t           | v                 |     |         |
    +---------------------------+-------------+-------------------+-----+---------+
    | Weibull(lamda, k)         | exponweib   | lamda, k          |     |         |
    +---------------------------+-------------+-------------------+-----+---------+

    Thus, each distribution above would have the same call signature::

        >>> import scipy.stats as ss
        >>> ss.your_dist_here(args,loc=loc,scale=scale)

    Convenient constructors have been created to make assigning these
    distributions easier. They follow the parameter notation found in the
    respective Wikipedia articles:

    +---------------------------+---------------------------------------------------------------+
    | MCERP Distibution         | Wikipedia page                                                |
    +===========================+===============================================================+
    | N(mu, sigma)              | http://en.wikipedia.org/wiki/Normal_distribution              |
    +---------------------------+---------------------------------------------------------------+
    | U(a, b)                   | http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)|
    +---------------------------+---------------------------------------------------------------+
    | Exp(lamda, [mu])          | http://en.wikipedia.org/wiki/Exponential_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Gamma(k, theta)           | http://en.wikipedia.org/wiki/Gamma_distribution               |
    +---------------------------+---------------------------------------------------------------+
    | Beta(alpha, beta, [a, b]) | http://en.wikipedia.org/wiki/Beta_distribution                |
    +---------------------------+---------------------------------------------------------------+
    | LogN(mu, sigma)           | http://en.wikipedia.org/wiki/Log-normal_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | X2(df)                    | http://en.wikipedia.org/wiki/Chi-squared_distribution         |
    +---------------------------+---------------------------------------------------------------+
    | F(dfn, dfd)               | http://en.wikipedia.org/wiki/F-distribution                   |
    +---------------------------+---------------------------------------------------------------+
    | Tri(a, b, c)              | http://en.wikipedia.org/wiki/Triangular_distribution          |
    +---------------------------+---------------------------------------------------------------+
    | T(df)                     | http://en.wikipedia.org/wiki/Student's_t-distribution         |
    +---------------------------+---------------------------------------------------------------+
    | Weib(lamda, k)            | http://en.wikipedia.org/wiki/Weibull_distribution             |
    +---------------------------+---------------------------------------------------------------+


    Thus, the following are equivalent::

        >>> x = uv([10, 1, 0, 3, 0, 15, 0, 105])
        >>> x = uv(rv=ss.norm(loc=10, scale=1))
        >>> x = N(10, 1)

    Examples
    --------
    Using the first eight distribution moments::

        >>> x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])  # normally distributed
        >>> x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])  # normally distributed
        >>> x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833]) # exp. distributed
        >>> Z = (x1*x2**2)/(15*(1.5 + x3))
        >>> Z
        uv(1176.45, 99699.6822919, 0.708013052954, 6.16324345122)

    The result shows the mean, variance, and standardized skewness and kurtosis
    of the output variable Z.

    Same example, but now using ``scipy.stats`` objects::

        >>> import scipy.stats as ss
        >>> x1 = uv(rv=ss.norm(loc=24, scale=1))  # normally distributed
        >>> x2 = uv(rv=ss.norm(loc=37, scale=4))  # normally distributed
        >>> x3 = uv(rv=ss.expon(scale=0.5))  # exponentially distributed

    Or using the convenient distribution constructors::

        >>> x1 = N(24, 1)
        >>> x2 = N(37, 4)
        >>> x3 = Exp(2)

    The results may be slightly different from using the moments manually since
    moment calculations can suffer from numerical errors during the integration
    of the expectation equations above, but they will be close enough.

    Basic math operations may be applied to distributions, where all
    statistical calculations are performed using method of moments. Built-in
    trig-, logarithm-, etc. functions should be used when possible since they
    support both scalar values and uncertain objects.

    At any time, the 8 standardized moments of variables (or the 4 that result
    from calculations) can be retrieved using::

        >>> x1.moments()
        [24.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0]

    Or any moment can be accessed directly by specifying its index::

        >>> Z.moments(1)  # variance
        99699.6822919

    Important
    ---------

    One final thing to note is that some answers suffer from the use of a
    second-order approximation to the method of moment equations. For example,
    the equation f(x) = x*sin(x) has this issue::

        >>> x = N(0, 1)  # standard normal distribution
        >>> x*sin(x)
        uv(1.0, 2.0, 2.82842712475, 15.0)

    This is the precise answer for f(x) = x**2, which just so happens to be the
    second-order Taylor series approximation of x*sin(x). The correct answer
    for [mean,variance,skewness,kurtosis] here can be calculated by::

        >>> mu = 0.0
        >>> sigma = 1.0
        >>> n = ss.norm(loc=mu, scale=sigma)
        >>> rm = [n.dist.expect(lambda x: (x*math.sin(x))**k, loc=mu,
        ...       scale=sigma) for k in (1, 2, 3, 4)]
        >>> cm = raw2central(rm)
        >>> mean = rm[0]
        >>> var = cm[1]
        >>> std = var**0.5
        >>> skew = cm[2]/std**3
        >>> kurt = cm[3]/std**4
        >>> [mean, var, skew, kurt]
        [0.6065306597, 0.3351234837, 0.6539519888, 2.5584134397]

    Thus, care should be taken to make sure that the equations used are
    effectively quadratic within the respective input variable distribution
    ranges or you will see approximation errors like the example above.

    """

    def __init__(self, moments=[], rv=None, tag=None):
        assert not all([not moments, not rv]), (
            "Either the moments must be "
            + 'put in manually or a "rv_continuous" object from the '
            + '"scipy.stats" module must be supplied'
        )

        if rv is not None:
            loc = rv.kwds.get("loc", 0.0)
            scale = rv.kwds.get("scale", 1.0)
            shape = rv.args

            mn = rv.mean()
            sd = rv.std()

            if shape:
                assert rv.dist.numargs >= 1, (
                    "The distribution provided "
                    + "doesn't support a 'shape' parameter"
                )
                expect = lambda k: rv.dist.expect(
                    lambda x: x**k, args=shape, loc=loc, scale=scale
                )
                raw_moments = [expect(k) for k in range(1, 9)]
                moments = raw2central(list(raw_moments))
                for k in range(2, 8):
                    moments[k] = moments[k] / sd ** (k + 1)
            else:
                assert rv.dist.numargs == 0, (
                    "The distribution provided "
                    + "requires a third 'shape' parameter"
                )
                expect = lambda k: rv.dist.expect(lambda x: x**k)
                raw_moments = [expect(k) for k in range(1, 9)]
                moments = raw2central(list(raw_moments))
            moments[0] = mn  # mean
            moments[1] = sd**2  # variance
            self._dist = rv
        else:
            self._dist = None
        # Ensure moments are floats, not ints, and not used as array shape or index
        self._moments = [float(m) for m in moments]
        self.x = self._moments[
            0
        ]  # Set x to mean value for arithmetic propagation
        self._lc = {}
        self._qc = {}
        self._cqc = {}
        self.tag = tag

    def __hash__(self):
        return id(self)

    @property
    def mean(self):
        return self.moments(0)

    @property
    def var(self):
        return self.moments(1)

    @property
    def std(self):
        return self.var**0.5

    @property
    def skew(self):
        return self.moments(2)

    @property
    def kurt(self):
        return self.moments(3)

    def moments(self, idx=None):
        if idx is not None and idx < len(self._moments):
            return self._moments[idx]
        else:
            return self._moments

    def set_mean(self, mn):
        """
        Modify the first moment via the mean
        """
        self._moments[0] = mn

    def set_std(self, sd):
        """
        Modify the second moment via the standard deviation
        """
        self._moments[1] = sd**2

    def set_var(self, vr):
        """
        Modify the second moment via the variance
        """
        self._moments[1] = vr

    def set_skew(self, sk):
        """
        Modify the third moment via the standardized skewness coefficient
        """
        self._moments[2] = sk

    def set_kurt(self, kt):
        """
        Modify the fourth moment via the standardized kurtosis coefficient
        """
        self._moments[3] = kt

    def set_moments(self, m):
        """
        Modify the first eight moments of the UncertainVariable's distribution
        """
        assert len(m) == 8, "Input moments must include eight values"
        self._moments = [float(val) for val in m]

    if matplotlib_installed:

        def plot(self, vals=None, **kwargs):
            """Plot the distribution of the input variable.

            NOTE: This requires defining the input using a distribution from
            the ``scipy.stats`` module.

            """
            if self._dist is not None:
                if vals is None:
                    low = self._dist.ppf(0.0001)
                    high = self._dist.ppf(0.9999)
                else:
                    low = min(vals)
                    high = max(vals)
                vals = np.linspace(low, high, 500)
                plt.plot(vals, self._dist.pdf(vals), **kwargs)
                # plt.title(repr(self))
                plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)
                plt.show()
            else:
                raise NotImplementedError(
                    "Cannot determine a distribution's "
                    + "pdf only by its moments (yet). Please use a scipy "
                    + "distribution if you want to plot."
                )


uv = UncertainVariable  # a nicer form for the user

###############################################################################
# Define some convenience constructors for common statistical distributions.
# Hopefully these are a little easier/more intuitive to use than the
# scipy.stats.distributions.
###############################################################################


def n(mu, sigma, tag=None):
    """
    A Normal (or Gaussian) random variate

    Parameters
    ----------
    mu : scalar
        The mean value of the distribution
    sigma : scalar
        The standard deviation (must be positive and non-zero)
    """
    assert sigma > 0, "Sigma must be positive"
    return uv(rv=ss.norm(loc=mu, scale=sigma), tag=tag)


###############################################################################


def u(a, b, tag=None):
    """
    A Uniform random variate

    Parameters
    ----------
    a : scalar
        Lower bound of the distribution support.
    b : scalar
        Upper bound of the distribution support.
    """
    assert a < b, "Lower bound must be less than the upper bound"
    return uv(rv=ss.uniform(loc=a, scale=b - a), tag=tag)


###############################################################################


def exp(lamda, tag=None):
    """
    An Exponential random variate

    Parameters
    ----------
    lamda : scalar
        The inverse scale (as shown on Wikipedia), FYI: mu = 1/lamda.
    """
    return uv(rv=ss.expon(scale=1.0 / lamda), tag=tag)


###############################################################################


def gamma(k, theta, tag=None):
    """
    A Gamma random variate

    Parameters
    ----------
    k : scalar
        The shape parameter (must be positive and non-zero)
    theta : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert k > 0 and theta > 0, "Gamma parameters must be greater than zero"
    return uv(rv=ss.gamma(k, scale=theta), tag=tag)


###############################################################################


def beta(alpha, beta, a=0, b=1, tag=None):
    """
    A Beta random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    Optional
    --------
    a : scalar
        Lower bound of the distribution support (default=0)
    b : scalar
        Upper bound of the distribution support (default=1)
    """
    assert alpha > 0 and beta > 0, "Shape parameters must be greater than zero"
    return uv(rv=ss.beta(alpha, beta, loc=a, scale=b - a), tag=tag)


###############################################################################


def logn(mu, sigma, tag=None):
    """
    A Log-Normal random variate

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert sigma > 0, "Sigma must be positive"
    return uv(rv=ss.lognorm(sigma, loc=mu), tag=tag)


###############################################################################


def chi2(df, tag=None):
    """
    A Chi-Squared random variate

    Parameters
    ----------
    df : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert isinstance(df, int) and df > 1, "DF must be an int greater than 1"
    return uv(rv=ss.chi2(df), tag=tag)


###############################################################################


def f(d1, d2, tag=None):
    """
    An F (fisher) random variate

    Parameters
    ----------
    d1 : int
        Numerator degrees of freedom
    d2 : int
        Denominator degrees of freedom
    """
    assert isinstance(d1, int) and d1 > 1, "d1 must be an int greater than 1"
    assert isinstance(d2, int) and d2 > 1, "d2 must be an int greater than 1"
    return uv(rv=ss.f(d1, d2), tag=tag)


###############################################################################


def tri(a, b, c, tag=None):
    """
    A triangular random variate

    Parameters
    ----------
    a : scalar
        Lower bound of the distribution support (default=0)
    b : scalar
        Upper bound of the distribution support (default=1)
    c : scalar
        The location of the triangle's peak (a <= c <= b)
    """
    assert a <= c <= b, "peak must lie in between low and high"
    return uv(rv=ss.triang(c, loc=a, scale=b - a), tag=tag)


###############################################################################


def t(v, tag=None):
    """
    A Student-T random variate

    Parameters
    ----------
    v : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert isinstance(v, int) and v > 1, "v must be an int greater than 1"
    return uv(rv=ss.t(v), tag=tag)


###############################################################################


def weib(lamda, k, tag=None):
    """
    A Weibull random variate

    Parameters
    ----------
    lamda : scalar
        The scale parameter
    k : scalar
        The shape parameter
    """
    assert lamda > 0 and k > 0, (
        "Weibull scale and shape parameters must be greater than zero"
    )
    return uv(rv=ss.exponweib(lamda, k), tag=tag)


###############################################################################


def raw2central(v):
    """Convert raw moments (1 to len(v)) to central moments"""

    def nci(n, i):
        return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))

    v = [1] + v
    central_moments = []
    for k in range(len(v)):
        val = 0.0

        # use the recursion definition to transform
        for j in range(k + 1):
            val += (-1) ** j * nci(k, j) * v[k - j] * v[1] ** j
        central_moments.append(val)

    return central_moments[1:]


def covariance_matrix(nums_with_uncert):
    """
    Calculate the covariance matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    cov_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> covariance_matrix([x, y, z])
        [[ 0.01  0.    0.01]
         [ 0.    0.01  0.02]
         [ 0.01  0.02  0.05]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = []
    for i1, expr1 in enumerate(ufuncs):
        derivatives1 = expr1._lc  # Optimization
        vars1 = set(derivatives1)
        coefs_expr1 = []
        for i2, expr2 in enumerate(ufuncs[: i1 + 1]):
            derivatives2 = expr2._lc  # Optimization
            coef = 0.0
            for v in vars1.intersection(derivatives2):
                # v is a variable common to both numbers with
                # uncertainties:
                coef += derivatives1[v] * derivatives2[v] * v.var
            coefs_expr1.append(coef)
        cov_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for i, covariance_coefs in enumerate(cov_matrix):
        covariance_coefs.extend(
            cov_matrix[j][i] for j in range(i + 1, len(cov_matrix))
        )

    return cov_matrix


def correlation_matrix(nums_with_uncert):
    """
    Calculate the correlation matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    corr_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> correlation_matrix([x, y, z])
        [[ 1.          0.          0.4472136 ]
         [ 0.          1.          0.89442719]
         [ 0.4472136   0.89442719  1.        ]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = covariance_matrix(ufuncs)
    corr_matrix = []
    for i1, expr1 in enumerate(ufuncs):
        row_data = []
        for i2, expr2 in enumerate(ufuncs):
            row_data.append(cov_matrix[i1][i2] / expr1.std / expr2.std)
        corr_matrix.append(row_data)
    return corr_matrix
