# src/autora/theorist/nuts/theorist.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .config import NutsConfig
from .fitness import fitness as fitness_fn
from .genetic.evolve import evolve
from .audit import Logger
from .format import format_equation

class NutsTheorist:
    def __init__(self, config: NutsConfig | None = None):
        self.config = config or NutsConfig()
        self._best_tree = None
        self._best_consts = None
        self._fval = None
        self._var_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, var_names: list[str]) -> "NutsTheorist":
        # Prepare data dict for fast eval
        Xdict = {name: X[:, i] for i, name in enumerate(var_names)}
        self._var_names = var_names

        # TODO: initial population creation (grow function, constants init)
        pop = []           # list[Node]
        consts_list = []   # list[dict]

        logger = Logger(self.config.csv_path, self.config.keep_top_k)
        grow_fn = None     # TODO: pass your ramped-half-and-half grow function

        best_tree, best_consts, fval = evolve(
            pop, consts_list, Xdict, y, self.config,
            lambda t, Xd, yy, c: fitness_fn(t, Xd, yy, c, self.config),
            grow_fn,
            logger
        )

        self._best_tree, self._best_consts, self._fval = best_tree, best_consts, fval
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._best_tree is None:
            raise RuntimeError("Call fit() first.")
        Xdict = {name: X[:, i] for i, name in enumerate(self._var_names)}
        return self._best_tree.eval(Xdict, self._best_consts)

    def print_eqn(self) -> str:
        if self._best_tree is None:
            return "<unfitted NutsTheorist>"
        return format_equation(self._best_tree)

    # Optional: expose equation & constants for benchmarking exports
    @property
    def best(self) -> Dict[str, Any]:
        return {
            "fval": self._fval,
            "equation": self.print_eqn(),
            "constants": self._best_consts,
            "symbols": self._best_tree.symbol_count() if self._best_tree else None,
        }
