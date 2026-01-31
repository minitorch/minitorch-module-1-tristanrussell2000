from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, runtime_checkable, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation
def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    val_minus = [val for val in vals]
    val_minus[arg] -= epsilon
    val_plus = [val for val in vals]
    val_plus[arg] += epsilon
    return (f(*val_plus) - f(*val_minus)) / (2 * epsilon)


variable_count = 1

class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    l = []
    visited = set()
    def visit(v: Variable):
        if v.unique_id in visited:
            return
        for child in v.parents:
            visit(child)
        l.insert(0, v)
        visited.add(v.unique_id)
    visit(variable)
    return l

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted = topological_sort(variable)
    print("start", sorted, deriv)
    scalar_derivs: Dict[int, float] = {s.unique_id: 0.0 for s in sorted}
    scalar_derivs[variable.unique_id] = deriv
    for v in sorted:
        print("isleaf", v.is_leaf())
        d = scalar_derivs[v.unique_id]
        if v.is_leaf(): v.accumulate_derivative(d)
        for parent, coeff in v.chain_rule(d):
            scalar_derivs[parent.unique_id] += coeff


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
