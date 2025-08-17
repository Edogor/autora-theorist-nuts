# src/autora/theorist/nuts/format.py
def format_equation(tree) -> str:
    # Convert tree to infix string using pset printable symbols
    # TODO: implement properly
    return f"<eqn symbols={tree.symbol_count()}>"
