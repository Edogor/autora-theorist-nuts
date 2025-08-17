# src/autora/theorist/nuts/audit.py
from dataclasses import dataclass
from typing import List, Optional
import csv

@dataclass
class ExportRow:
    generation: int
    mse: float
    symbols: int
    equation: str
    c_values: dict

class Logger:
    def __init__(self, csv_path: Optional[str], keep_top_k: int):
        self.csv_path = csv_path
        self.keep_top_k = keep_top_k
        self.rows: List[ExportRow] = []
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["generation","mse","symbols","equation","constants"])

    def on_generation(self, gen, pop, consts_list, fitnesses):
        ranked = sorted(zip(pop, consts_list, fitnesses), key=lambda t: t[2])[: self.keep_top_k]
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for t, c, fit in ranked:
                    writer.writerow([gen, fit, t.symbol_count(), str(t), c])
