# src/autora/theorist/nuts/numeric.py
import numpy as np

def clamp(x, lo, hi):
    return np.clip(x, lo, hi)

def safe_pow(a, b, eps=1e-9, lo=-1e6, hi=1e6):
    a = np.clip(a, eps, None)   # avoid 0^neg, negative bases, etc.
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.power(a, b)
    return clamp(out, lo, hi)

def safe_exp(x, lo=-1e6, hi=1e6):
    x = np.clip(x, -40, 40)     # hard cap to prevent overflow
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.exp(x)
    return clamp(out, lo, hi)

def safe_div(a, b, eps=1e-9):
    return a / (np.sign(b)*np.maximum(np.abs(b), eps))

def safe_log(x, eps=1e-9):
    return np.log(np.maximum(x, eps))
