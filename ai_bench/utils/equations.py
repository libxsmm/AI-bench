import ast
from collections.abc import Mapping
from collections.abc import Sequence
import operator as op


def _flatten_numeric_sequence(values: Sequence[object]) -> list[int | float]:
    """Flatten nested helper arguments into a numeric sequence.
    Args:
        values: Scalar and sequence values passed to a helper
    Returns:
        Flat list of numeric values
    """
    flattened: list[int | float] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, str):
            flattened.extend(_flatten_numeric_sequence(value))
        elif isinstance(value, (int, float)):
            flattened.append(value)
        else:
            raise TypeError("Helper arguments must resolve to numbers or sequences")
    return flattened


def adjacent_prod_sum(*parts: object) -> int | float:
    """Sum products of adjacent values across scalar and list inputs.
    Args:
        parts: Scalar values or sequences of values
    Returns:
        Sum of pairwise products across adjacent values
    """
    values = _flatten_numeric_sequence(parts)
    if len(values) < 2:
        return 0
    return sum(left * right for left, right in zip(values, values[1:]))


# List of supported operations.
_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# Whitelisted helpers that formula expressions may call.
_ALLOWED_FUNCS = {
    "adjacent_prod_sum": adjacent_prod_sum,
}


def eval_eq(equation: str, variables: Mapping[str, object] | None = None) -> float:
    """Evaluate arithmetic expression.
    Args:
        equation: Equation string
        variables: Optional variable bindings used by the expression
    Returns:
        Numerical result
    """
    return eval_ast(ast.parse(equation, mode="eval").body, variables or {})


def eval_ast(node: ast.AST, variables: Mapping[str, object]) -> float | list[object]:
    """Evaluate arithmetic equation AST.
    Args:
        node: AST node to evaluate
        variables: Variable bindings used by the expression
    Returns:
        Numerical result
    """
    match node:
        case ast.Constant(value) if isinstance(value, (int, float)):
            return value
        case ast.List(elts=elts, ctx=ast.Load()) | ast.Tuple(elts=elts, ctx=ast.Load()):
            return [eval_ast(elt, variables) for elt in elts]
        case ast.Name(id=name):
            if name not in variables:
                raise NameError(f"Unknown variable: {name}")
            return variables[name]
        case ast.Subscript(value, slice, ctx=ast.Load()):
            container = eval_ast(value, variables)
            index = eval_ast(slice, variables)
            if not isinstance(container, Sequence) or isinstance(container, str):
                raise TypeError("Subscript target must be a sequence")
            if not isinstance(index, int):
                raise TypeError("Subscript index must be an integer")
            return container[index]
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
            if name not in _ALLOWED_FUNCS:
                raise TypeError(f"Unsupported function: {name}")
            evaluated_args = [eval_ast(arg, variables) for arg in args]
            return _ALLOWED_FUNCS[name](*evaluated_args)
        case ast.UnaryOp(op, operand) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](eval_ast(operand, variables))
        case ast.BinOp(left, op, right) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](
                eval_ast(left, variables), eval_ast(right, variables)
            )
        case _:
            raise TypeError("Unsupported token")
