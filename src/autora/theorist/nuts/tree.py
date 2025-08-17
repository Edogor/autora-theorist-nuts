# src/autora/theorist/nuts/tree.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Dict, Callable
import numpy as np
from .pset import PRIMITIVES

@dataclass
class Node:
    op: str                       # key in PRIMITIVES or "var" or "const"
    children: List["Node"] = None
    value: str | None = None      # variable name or constant symbol

    def arity(self) -> int:
        if self.op in ("var", "const"):
            return 0
        return PRIMITIVES[self.op][1]

    def symbol_count(self) -> int:
        if self.op in ("var", "const"):
            return 1
        return 1 + sum(c.symbol_count() for c in self.children or [])

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def eval(self, X: Dict[str, np.ndarray], consts: Dict[str, float]) -> np.ndarray:
        if self.op == "var":
            return X[self.value]
        if self.op == "const":
            return np.full_like(next(iter(X.values())), fill_value=consts[self.value], dtype=float)
        fn, _, _ = PRIMITIVES[self.op]
        args = [ch.eval(X, consts) for ch in self.children]
        return fn(*args)
