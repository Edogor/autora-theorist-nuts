# src/autora/theorist/nuts/config.py
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class NutsConfig:
    population_size: int = 200
    generations: int = 50
    tournament_k: int = 7
    elitism: int = 5
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    constant_mutation_sigma: float = 0.1

    # constraints
    max_depth: int = 10
    max_symbols: int = 40                   # challenge limit
    ban_nested_safe_exp: bool = True

    # numerics
    eps: float = 1e-9
    clip_lo: float = -1e6
    clip_hi: float = 1e6

    # terminals
    constant_symbols: Sequence[str] = ("c1", "c2", "c3")

    # logging/export
    keep_top_k: int = 20
    csv_path: Optional[str] = None
