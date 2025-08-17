# src/autora/theorist/nuts/fitness.py
import numpy as np
from .constraints import check_bans, enforce_symbol_budget

def mse(y_true, y_pred):
    diff = y_true - y_pred
    return float(np.mean(diff * diff))

def fitness(tree, X, y, consts, config):
    # hard constraints first
    if not enforce_symbol_budget(tree, config.max_symbols):
        return np.inf
    if config.ban_nested_safe_exp and not check_bans(tree):
        return np.inf

    y_hat = tree.eval(X, consts)
    base = mse(y, y_hat)
    # small complexity pressure helps the search
    penalty = 1e-3 * tree.symbol_count()
    return base + penalty
