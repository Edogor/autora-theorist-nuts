# src/autora/theorist/nuts/constraints.py
def enforce_symbol_budget(tree, max_symbols: int) -> bool:
    return tree.symbol_count() <= max_symbols

def check_bans(tree) -> bool:
    # Example: ban nested safe_exp
    def _walk(n, depth_exp=0):
        if n.op == "exp":
            depth_exp += 1
            if depth_exp > 1:
                return False
        for c in (n.children or []):
            if not _walk(c, depth_exp):
                return False
        return True
    return _walk(tree)
