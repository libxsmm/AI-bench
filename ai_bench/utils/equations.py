import ast
import operator as op

# List of supported operations
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def eval_eq(equation: str) -> float:
    """Evaluate arithmetic expression.
    Args:
        equation: Equation string
    Returns:
        Numerical result
    """
    return eval_ast(ast.parse(equation, mode="eval").body)


def eval_ast(node: ast.AST) -> float:
    """Evaluate arithmetic equation AST.
    Args:
        node: AST node to evaluate
    Returns:
        Numerical result
    """
    match node:
        case ast.Constant(value) if isinstance(value, (int, float)):
            return value
        case ast.UnaryOp(op, operand) if type(op) in operators:
            return operators[type(op)](eval_ast(operand))
        case ast.BinOp(left, op, right) if type(op) in operators:
            return operators[type(op)](eval_ast(left), eval_ast(right))
        case _:
            raise TypeError("Unsupported token")
