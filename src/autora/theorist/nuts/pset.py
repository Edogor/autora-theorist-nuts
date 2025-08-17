# src/autora/theorist/nuts/pset.py
import numpy as np
from .numeric import safe_pow, safe_exp, safe_div, safe_log

PRIMITIVES = {
    "add": (lambda a, b: a + b, 2, "+"),
    "sub": (lambda a, b: a - b, 2, "-"),
    "mul": (lambda a, b: a * b, 2, "*"),
    "div": (safe_div, 2, "/"),
    "pow": (safe_pow, 2, "^"),
    "exp": (safe_exp, 1, "safe_exp"),
    "log": (safe_log, 1, "log+"),
    "sin": (np.sin, 1, "sin"),
    "cos": (np.cos, 1, "cos"),
}
