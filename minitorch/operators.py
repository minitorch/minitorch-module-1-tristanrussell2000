"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Tuple

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def add(a: float, b: float) -> float:
    """Addition.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        Sum of a and b

    """
    return a + b


def neg(a: float) -> float:
    """Negation.

    Args:
    ----
        a: A float

    Returns:
    -------
        Negative of a

    """
    return -a


def id(a: float) -> float:
    """Identity.

    Args:
    ----
        a: A float

    Returns:
    -------
        a

    """
    return a


def mul(a: float, b: float) -> float:
    """Multiplication.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        Product of a and b

    """
    return a * b


def lt(a: float, b: float) -> bool:
    """Less than.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        True if a is less than b, False otherwise

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Equality.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        True if a is equal to b, False otherwise

    """
    return a == b


def max(a: float, b: float) -> float:
    """Maximum.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        The maximum of a and b

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Is close.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        True if a is close to b, False otherwise

    """
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid function.

    Args:
    ----
        x: A float

    Returns:
    -------
        The sigmoid of x

    """
    if x > 0:
        return 1 / (1 + math.pow(math.e, -x))
    else:
        return math.pow(math.e, x) / (1 + math.pow(math.e, x))


def log(a: float) -> float:
    """Logarithm.

    Args:
    ----
        a: A float

    Returns:
    -------
        The natural logarithm of a

    """
    return math.log(a)


def exp(x: float) -> float:
    """Exponentiation.

    Args:
    ----
        x: A float

    Returns:
    -------
        e raised to the power of x

    """
    return math.exp(x)


def log_back(a: float, b: float) -> float:
    """Backward pass for logarithm.

    Args:
    ----
        a: A float
        b: A float

    Returns:
    -------
        The derivative of log(a) with respect to a, multiplied by b

    """
    return b / a


def inv(x: float) -> float:
    """Inverse.

    Args:
    ----
        x: A float

    Returns:
    -------
        The inverse of x

    """
    return 1 / x


def inv_back(x: float, b: float) -> float:
    """Backward pass for inverse.

    Args:
    ----
        x: A float
        b: A float

    Returns:
    -------
        The derivative of 1/x with respect to x, multiplied by b

    """
    return -b / math.pow(x, 2)


def relu(x: float) -> float:
    """Rectified Linear Unit (ReLU).

    Args:
    ----
        x: A float

    Returns:
    -------
        x if x > 0, otherwise 0

    """
    if x <= 0:
        return 0
    return x


def relu_back(x: float, b: float) -> float:
    """Backward pass for ReLU.

    Args:
    ----
        x: A float
        b: A float

    Returns:
    -------
        b if x > 0, otherwise 0

    """
    if x <= 0:
        return 0
    else:
        return b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable, l1: Iterable) -> Iterable:
    for n in l1:
        yield f(n)


def zipWith[T, G](l1: Iterable[T], l2: Iterable[G]) -> Iterable[Tuple[T, G]]:
    it1 = iter(l1)
    it2 = iter(l2)
    while True:
        try:
            n1 = next(it1)
            n2 = next(it2)
        except StopIteration:
            return
        yield (n1, n2)


def reduce[T, G](op: Callable[[T, G], G], init: G, l: Iterable[T]) -> G:
    build = init
    for n in l:
        build = op(n, build)
    return build


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    return reduce(lambda z, sum: sum + [z[0] + z[1]], [], zipWith(l1, l2))


def negList(l: Iterable) -> Iterable:
    return map(lambda x: -x, l)


def sum(l1: Iterable[float]) -> float:
    return reduce(add, 0, l1)


def prod(l1: Iterable[float]) -> float:
    return reduce(mul, 1, l1)
