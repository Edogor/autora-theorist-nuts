# NutsTheorists — Island Model + Subtree Splitting (full drop-in)

from typing import List, Optional, Dict
import random
import copy
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from itertools import product

warnings.filterwarnings('ignore', category=RuntimeWarning)

class NutsTheorists(BaseEstimator):
    """
    Genetic 'Theorist' with:
      - Subtree Splitting of top trees (middle/quantiles/random)
      - Island Model ('Biomes') with independent evolution + migration
      - Vectorized constant fitting + random sampling
      - Manual stop during fit() with Top-5 snapshot
      - Time-budget stop (minutes) for fit()
    """

    # ---------- Biome container ----------
    class _Biome:
        def __init__(self, params: Dict, population: List):
            self.params = params              # per-biome overrides (mutation_rate, max_depth, ...)
            self.population = population      # list of trees
            self.hof: List = []               # Hall of fame per biome (tuples (tree, mse, consts))
            self.best = (None, float("inf"), {})  # (tree, mse, consts)

    # ---------- Init ----------
    def __init__(
        self,
        population_size=300,
        n_generation=30,
        mutation_rate=0.2,
        tournament_size=50,
        early_stopping_rounds=5,
        complexity_penalty=0.02,
        n_constants=1,
        verbose=True,
        elitism=20,
        constant_grid=None,
        # --- Subtree splitting controls ---
        enable_splitting=True,
        split_n_best=20,
        split_m_per_tree=3,
        split_strategy="quantiles",  # "quantiles" | "middle" | "random"
        split_min_depth=2,
        # --- Island model / Biomes ---
        n_biomes=3,
        biome_configs=None,         # list of dicts overriding per-biome params
        migration_interval=5,
        migration_size=20,
        migration_policy="ring_topk",  # "ring_topk" | "ring_random" | "new_biome_mix"
        max_biomes=6,
        # --- Constant search (NEW) ---
        constant_search: str = "random",  # "random" | "grid"
        constant_samples: int = 100,      # random samples per tree
    ):
        # base GA
        self.population_size = population_size
        self.n_generation = n_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.early_stopping_rounds = early_stopping_rounds
        self.complexity_penalty = complexity_penalty
        self.n_constants = n_constants
        self.elitism = elitism
        self.verbose = verbose

        self.constant_grid = constant_grid or np.linspace(-3, 3.0, 20)
        self.constant_search = constant_search
        self.constant_samples = constant_samples
        self._expr_cache: Dict[str, object] = {}  # compile-cache

        self.UNARY_OPS = ['np.log', 'np.exp']
        self.UNARY_OPS = ['np.log', 'np.exp', 'sigmoid']  # <<< ADDED 'sigmoid'

        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']
        self.hall_of_fame: List = []

        # splitting
        self.enable_splitting = enable_splitting
        self.split_n_best = split_n_best
        self.split_m_per_tree = split_m_per_tree
        self.split_strategy = split_strategy
        self.split_min_depth = split_min_depth

        # biome / islands
        self.n_biomes = n_biomes
        self.biome_configs = biome_configs or []
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.migration_policy = migration_policy
        self.max_biomes = max_biomes

        # --- Manual & time stop control ---
        self._stop_requested = False
        self._deadline: Optional[float] = None  # absolute time.monotonic() seconds
        self.time_limit_min: Optional[float] = None  # default per instance
        self.last_topk: List = []

    # --- Manual/time stop API ---
    def request_stop(self):
        """Von außen aufrufen, um das Training sauber zu beenden."""
        self._stop_requested = True

    def cancel_stop(self):
        """Stop-Flag zurücksetzen (für nächsten Fit-Lauf)."""
        self._stop_requested = False

    def set_time_limit(self, minutes: Optional[float]):
        """
        Standard-Zeitlimit (in Minuten) für kommende fit()-Aufrufe setzen.
        Beispiel: set_time_limit(1.5)  # 1.5 Minuten
        """
        self.time_limit_min = float(minutes) if minutes is not None else None

    def clear_time_limit(self):
        """Gesetztes Standard-Zeitlimit entfernen."""
        self.time_limit_min = None

    def time_remaining(self) -> Optional[float]:
        """
        Restzeit in Sekunden bis zum Zeitlimit; None falls kein Limit aktiv.
        (Nur während fit() sinnvoll.)
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def _start_deadline(self, time_limit_min: Optional[float]):
        """
        Interne Deadline starten. Priorität:
        - explizites fit(..., time_limit_min=...) für diesen Lauf
        - ansonsten self.time_limit_min (Default für Instanz)
        """
        limit = time_limit_min if time_limit_min is not None else self.time_limit_min
        self._deadline = (time.monotonic() + float(limit) * 60.0) if (limit is not None) else None

    def _check_time(self) -> bool:
        """
        Prüft Zeitlimit; setzt Stop-Flag und gibt True zurück, wenn abgelaufen.
        """
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._stop_requested = True
            return True
        return False

    def _pretty_equation(self, tree, constants):
        """Nur zur Anzeige: Konstanten (falls vorhanden) in die Gleichung einsetzen."""
        s = self._tree_translate(tree) if tree is not None else ""
        for k, v in (constants or {}).items():
            s = s.replace(k, f"{float(v):.6g}")
        return s

    def topk(self, n: int = 5):
        """
        Top-k Kandidaten (Snapshot, z.B. nach manuellem Stopp/Zeitlimit).
        Rückgabe: Liste von Dicts {eq, mse, consts}.
        """
        src = self.last_topk or sorted(self.hall_of_fame, key=lambda x: x[1])[:n]
        out = []
        for t, mse, consts in src[:n]:
            out.append({
                "eq": self._pretty_equation(t, consts),
                "mse": float(mse),
                "consts": dict(consts)
            })
        return out

    # ---------- Tree basics ----------
    def _init_constants_and_terminals(self, var_names):
        self.var_names = var_names
        self.constant_names = [f"c{i+1}" for i in range(self.n_constants)]
        self.TERMINALS = self.constant_names + self.var_names
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

    def _create_random_tree(self, max_depth=3):
        if max_depth == 0 or random.random() < 0.2:
            return random.choice(self.TERMINALS)
        op = random.choice(self.UNARY_OPS + self.BINARY_OPS)
        if op in self.UNARY_OPS:
            return [op, self._create_random_tree(max_depth - 1)]
        return [op, self._create_random_tree(max_depth - 1), self._create_random_tree(max_depth - 1)]

    def _tree_translate(self, tree):
        if isinstance(tree, str):
            return tree
        op, *args = tree
        translated = [self._tree_translate(arg) for arg in args]
        if op == 'np.exp':
            return f"exp({translated[0]})"
        elif op == 'np.log':
            return f"log({translated[0]})"
        elif op == 'np.power':
            return f"({translated[0]} ^ {translated[1]})"
        return f"({translated[0]} {op} {translated[1]})"

    def _prepare_equation(self, eq_str, constants):
        eq_str = eq_str.replace('^', '**').replace('exp', 'np.exp').replace('log', 'np.log')
        for k, v in constants.items():
            eq_str = eq_str.replace(k, str(v))
        return eq_str

    # -------- Vectorized expression helpers (NEW) --------
    def _prepare_equation_template(self, eq_str: str) -> str:
        # wie _prepare_equation, aber lässt c1, c2, ... stehen
        return (eq_str
                .replace('^', '**')
                .replace('exp', 'np.exp')
                .replace('log', 'np.log'))

    def _compile_or_get(self, eq_template: str):
        code = self._expr_cache.get(eq_template)
        if code is None:
            code = compile(eq_template, "<expr>", "eval")
            self._expr_cache[eq_template] = code
        return code

    def _eval_expr_vectorized(self, code_obj, X_df: pd.DataFrame, constants: dict) -> np.ndarray:
        # Variablen als Arrays; Konstanten als Skalar-Bindings
        local_env = {name: X_df[name].values for name in self.var_names}
        local_env.update(constants)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            y = eval(code_obj, {"np": np}, local_env)
        y = np.asarray(y)
        if y.ndim == 0:  # konstante Funktion -> broadcast
            y = np.full(len(X_df), float(y))
        return y

    def _sample_constants(self, n: int, names: Optional[List[str]] = None) -> list:
        """
        Ziehe Zufallswerte nur für die angegebenen Konstantennamen (falls gesetzt).
        """
        names = list(names) if names is not None else list(self.constant_names)
        if not names:
            return [{}]
        low, high = -5.0, 5.0
        vals = np.random.uniform(low, high, size=(n, len(names)))
        return [dict(zip(names, row)) for row in vals]

    def _translate_tree_to_callable(self, eq_str, constants):
        prepared = self._prepare_equation(eq_str, constants)
        return eval(f"lambda {', '.join(self.var_names)}: {prepared}", {"np": np})

    def _count_symbols(self, tree):
        if isinstance(tree, str):
            return 1
        return 1 + sum(self._count_symbols(child) for child in tree[1:])

    def _is_constant_only(self, tree):
        if isinstance(tree, str):
            return tree in self.constant_names
        return all(self._is_constant_only(child) for child in tree[1:])

    # ---------- Fitness ----------
    def _evaluate_tree_mse(self, tree, conditions, observations):
        # konstante Doppelbewertungen vermeiden
        key = str(tree)
        if self._is_constant_only(tree):
            if key in self._seen_constant_only:
                return tree, float("inf"), {k: 1.0 for k in self.constant_names}
            else:
                self._seen_constant_only.add(key)

        symbol_count = self._count_symbols(tree)
        if symbol_count > 40:
            return tree, float("inf"), {k: 1.0 for k in self.constant_names}

        # Ausdruck -> Template -> compile (cached)
        eq_str = self._tree_translate(tree)                   # z. B. "(S ^ c1) + c2"
        eq_template = self._prepare_equation_template(eq_str) # z. B. "(S ** c1) + c2" mit np.*
        try:
            code = self._compile_or_get(eq_template)
        except Exception:
            return tree, float("inf"), {k: 1.0 for k in self.constant_names}

        # -> Welche Konstanten werden tatsächlich im Ausdruck referenziert?
        names_in_expr = set(getattr(code, "co_names", ()))  # Bytecode-Namen (np, S, c1, ...)
        used_constants = [c for c in self.constant_names if c in names_in_expr]

        y_true = observations.values.ravel()
        best_mse = float("inf")
        best_constants: Dict[str, float] = {}  # nur die wirklich benutzten

        # Fall A: Ausdruck hat KEINE Konstanten -> einmal vektorisiert auswerten
        if not used_constants:
            try:
                preds = self._eval_expr_vectorized(code, conditions, {})
                if np.any(~np.isfinite(preds)):
                    raise ValueError
                best_mse = mean_squared_error(y_true, preds.ravel())
                best_mse += self.complexity_penalty * symbol_count
            except Exception:
                best_mse = float("inf")
        else:
            # Fall B: Ausdruck nutzt Konstanten -> NUR diese durchsuchen
            if self.constant_search == "grid":
                vals = list(product(self.constant_grid, repeat=len(used_constants)))
                random.shuffle(vals)
                const_iter = (dict(zip(used_constants, v)) for v in vals)
            else:
                const_iter = self._sample_constants(self.constant_samples, names=used_constants)

            for constants in const_iter:
                try:
                    preds = self._eval_expr_vectorized(code, conditions, constants)
                    if np.any(~np.isfinite(preds)):
                        continue
                    mse = mean_squared_error(y_true, preds.ravel())
                    mse += self.complexity_penalty * symbol_count
                    if mse < best_mse:
                        best_mse = mse
                        best_constants = constants
                        # optional: strenger early-exit
                        if best_mse < 1e-12:
                            break
                except Exception:
                    continue

        mse = best_mse
        if self.verbose:
            print(f"Evaluated: {eq_str}, MSE: {mse:.5f}, constants: {best_constants}, complexity: {symbol_count}")

        if np.isfinite(mse):
            self.hall_of_fame.append((copy.deepcopy(tree), mse, best_constants))
            self.hall_of_fame.sort(key=lambda x: x[1])
            self.hall_of_fame = self.hall_of_fame[:10]

        return tree, mse, best_constants

    # ---------- GA ops ----------
    def _get_random_subtree(self, tree):
        candidates = []
        def collect(node, parent=None, idx=None, depth=0):
            if isinstance(node, list):
                candidates.append((node, parent, idx, depth))
                for i, child in enumerate(node[1:], 1):
                    collect(child, node, i, depth + 1)
        collect(tree)
        if not candidates:
            return tree, None, None
        node, parent, idx, _ = random.choices(candidates, weights=[d+1 for *_, d in candidates])[0]
        return node, parent, idx

    def _crossover(self, t1, t2):
        t1, t2 = copy.deepcopy(t1), copy.deepcopy(t2)
        n1, p1, i1 = self._get_random_subtree(t1)
        n2, p2, i2 = self._get_random_subtree(t2)
        if p1 and p2:
            p1[i1], p2[i2] = n2, n1
        return t1, t2

    def _mutate(self, tree, max_depth=3):
        def recurse(node):
            if not isinstance(node, list):
                return random.choice(self.TERMINALS) if random.random() < self.mutation_rate else node
            if random.random() < self.mutation_rate:
                return self._create_random_tree(max_depth)
            return [node[0]] + [recurse(child) for child in node[1:]]
        return recurse(copy.deepcopy(tree))

    def _tournament(self, population_scores, k=50):
        selected = random.sample(population_scores, min(len(population_scores), self.tournament_size * k))
        return sorted(selected, key=lambda x: x[1])[:k]

    def generate_next_generation(self, parents: List, max_depth=3):
        """
        parents: list of trees (e.g., top_k + split subtrees)
        """
        parents = copy.deepcopy(parents)
        if len(parents) < 2:
            return parents

        # Keep elites intact (up to available parents)
        new_pop = copy.deepcopy(parents[:min(self.elitism, len(parents))])

        # Fill remaining with crossover + mutation
        while len(new_pop) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            for child in self._crossover(p1, p2):
                mutated = self._mutate(child, max_depth)
                if self._count_symbols(mutated) <= 40:
                    try:
                        self._translate_tree_to_callable(self._tree_translate(mutated), self.best_c)
                        new_pop.append(mutated)
                    except:
                        continue
                if len(new_pop) >= self.population_size:
                    break
        return new_pop

    # ---------- Subtree splitting helpers ----------
    def _tree_depth(self, node):
        if isinstance(node, str):
            return 0
        return 1 + max(self._tree_depth(c) for c in node[1:])

    def _subtree_size(self, node):
        return self._count_symbols(node)

    def _collect_nodes_with_meta(self, tree):
        out = []
        def dfs(n, parent=None, idx=None, depth=0):
            if isinstance(n, list):
                op = n[0]
                out.append({"node": n, "parent": parent, "idx": idx,
                            "depth": depth, "size": self._subtree_size(n), "op": op})
                for i, child in enumerate(n[1:], 1):
                    dfs(child, n, i, depth+1)
        dfs(tree)
        return out

    def _split_node_to_children(self, node):
        if not isinstance(node, list):
            return []
        if len(node) == 2:        # unary
            return [copy.deepcopy(node[1])]
        elif len(node) == 3:      # binary
            return [copy.deepcopy(node[1]), copy.deepcopy(node[2])]
        return []

    def _pick_split_points(self, candidates, m, strategy):
        n = len(candidates)
        if n == 0:
            return []
        if strategy == "middle":
            mid = n // 2
            order = [mid]
            k = 1
            while len(order) < m and (mid-k >= 0 or mid+k < n):
                if mid-k >= 0: order.append(mid-k)
                if len(order) >= m: break
                if mid+k < n: order.append(mid+k)
                k += 1
            return [candidates[i] for i in order[:m]]

        if strategy == "quantiles":
            qs = [0.5, 0.25, 0.75, 0.125, 0.875, 0.0625, 0.9375]
            while len(qs) < m:
                qs.append(qs[-2] / 2)
                qs.append(1 - qs[-2])
            picks, used = [], set()
            for q in qs:
                i = int(round(q * (n - 1)))
                if i not in used:
                    picks.append(candidates[i])
                    used.add(i)
                if len(picks) >= m:
                    break
            return picks

        # random
        weights = [max(1, c["size"]) for c in candidates]
        total = sum(weights)
        probs = [w/total for w in weights]
        idxs = np.random.choice(np.arange(n), size=min(m, n), replace=False, p=probs)
        return [candidates[i] for i in idxs]

    def _valid_new_individual(self, tree):
        if self._count_symbols(tree) > 40:
            return False
        if self._is_constant_only(tree):
            return False
        try:
            self._translate_tree_to_callable(self._tree_translate(tree), self.best_c)
            return True
        except:
            return False

    def _split_top_trees(
        self,
        top_trees,
        n_best=None,
        m_per_tree=None,
        strategy=None,
        min_depth=None,
        max_new=100,
    ):
        if not self.enable_splitting:
            return []
        n_best = n_best if n_best is not None else self.split_n_best
        m_per_tree = m_per_tree if m_per_tree is not None else self.split_m_per_tree
        strategy = strategy or self.split_strategy
        min_depth = min_depth if min_depth is not None else self.split_min_depth

        new_individuals = []
        seen = set()

        for base_tree in top_trees[:n_best]:
            meta = self._collect_nodes_with_meta(base_tree)
            cands = [m for m in meta if isinstance(m["node"], list) and m["depth"] >= min_depth]
            cands.sort(key=lambda d: d["size"])

            split_points = self._pick_split_points(cands, m_per_tree, strategy)
            for sp in split_points:
                for ch in self._split_node_to_children(sp["node"]):
                    if self._valid_new_individual(ch):
                        key = str(ch)
                        if key not in seen:
                            new_individuals.append(ch)
                            seen.add(key)
                            if len(new_individuals) >= max_new:
                                return new_individuals
        return new_individuals

    # ---------- Biome helpers ----------
    def _default_biome_params(self, i: int) -> Dict:
        presets = [
            {"mutation_rate": 0.35, "complexity_penalty": 0.01, "max_depth": 6, "split_strategy": "random"},
            {"mutation_rate": 0.20, "complexity_penalty": 0.03, "max_depth": 6, "split_strategy": "quantiles"},
            {"mutation_rate": 0.15, "complexity_penalty": 0.05, "max_depth": 5, "split_strategy": "middle"},
        ]
        p = dict(presets[i % len(presets)])
        if i < len(self.biome_configs):
            p.update(self.biome_configs[i])
        return p

    def _init_biomes(self):
        biomes = []
        for i in range(self.n_biomes):
            params = self._default_biome_params(i)
            pop = [self._create_random_tree(params.get("max_depth", 6)) for _ in range(self.population_size)]
            biomes.append(self._Biome(params, pop))
        return biomes

    def _evolve_biome_one_gen(self, biome, X, y):
        # Temporarily override selected params for this biome
        saved = (self.mutation_rate, self.complexity_penalty, self.tournament_size)
        self.mutation_rate = biome.params.get("mutation_rate", self.mutation_rate)
        self.complexity_penalty = biome.params.get("complexity_penalty", self.complexity_penalty)
        self.tournament_size = biome.params.get("tournament_size", self.tournament_size)

        # Bewertet einzeln, damit wir Stop-/Zeit-Flag prüfen können
        pop_scores = []
        for t in biome.population:
            if self._check_time() or self._stop_requested:
                break
            pop_scores.append(self._evaluate_tree_mse(t, X, y))

        if not pop_scores:
            # Nichts bewertet -> sauber zurück
            self.mutation_rate, self.complexity_penalty, self.tournament_size = saved
            return float("inf")

        top_k = self._tournament(pop_scores)
        top_trees = [t for t, _, _ in top_k]

        # Subtree splitting per biome preferences
        if not (self._stop_requested or self._check_time()):
            splits = self._split_top_trees(
                top_trees,
                n_best=biome.params.get("split_n_best", self.split_n_best),
                m_per_tree=biome.params.get("split_m_per_tree", self.split_m_per_tree),
                strategy=biome.params.get("split_strategy", self.split_strategy),
                min_depth=biome.params.get("split_min_depth", self.split_min_depth),
                max_new=max(50, self.population_size // 5),
            )
            parents = top_trees + splits
            biome.population = self.generate_next_generation(parents, max_depth=biome.params.get("max_depth", 6))

        # Update per-biome best / HOF
        if self.hall_of_fame:
            best_tree, best_mse, best_c = min(self.hall_of_fame, key=lambda x: x[1])
            if best_mse < biome.best[1]:
                biome.best = (best_tree, best_mse, best_c)
            biome.hof = (biome.hof + self.hall_of_fame)[-10:]

        # restore global params
        self.mutation_rate, self.complexity_penalty, self.tournament_size = saved

        valid_mses = [s for _, s, _ in pop_scores if np.isfinite(s)]
        return np.mean(valid_mses) if valid_mses else float("inf")

    def _migrate(self, biomes: List["_Biome"]):
        if self.migration_policy.startswith("ring"):
            # i -> (i+1)
            for i, b in enumerate(biomes):
                if self.migration_policy == "ring_topk":
                    src = [t for t, _, _ in (b.hof or [])]
                    need = self.migration_size - len(src)
                    if need > 0 and len(b.population) > 0:
                        src += random.sample(b.population, min(len(b.population), need))
                else:
                    src = random.sample(b.population, min(len(b.population), self.migration_size))

                dst = biomes[(i + 1) % len(biomes)]
                for t in src:
                    if len(dst.population) == 0:
                        break
                    j = random.randrange(len(dst.population))
                    dst.population[j] = copy.deepcopy(t)

        elif self.migration_policy == "new_biome_mix" and len(biomes) < self.max_biomes:
            ranked = sorted(biomes, key=lambda b: b.best[1])
            donors = ranked[:2] if len(ranked) >= 2 else ranked
            seeds = []
            for b in donors:
                seeds += [t for t, _, _ in (b.hof or [])][:max(1, self.migration_size // max(1, len(donors)))]
            if not seeds:
                for b in donors:
                    if b.population:
                        seeds += random.sample(b.population, min(len(b.population), max(1, self.migration_size // max(1, len(donors)))))

            params = {"mutation_rate": 0.4, "complexity_penalty": 0.015, "max_depth": 6, "split_strategy": "random"}
            # Build new population by mutating seeds aggressively, fallback to random
            pop = []
            while len(pop) < self.population_size:
                base = random.choice(seeds) if seeds else None
                if base is None:
                    pop.append(self._create_random_tree(params["max_depth"]))
                else:
                    pop.append(self._mutate(base, params["max_depth"]))
            biomes.append(self._Biome(params, pop))

    # ---------- Fit / Predict ----------
    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame, time_limit_min: Optional[float] = None):
        # init
        self.cancel_stop()  # Stop-Flag für diesen Lauf zurücksetzen
        self._start_deadline(time_limit_min)  # Zeitlimit (falls gesetzt) aktivieren
        self.best_tree = None
        self.best_equation = None
        self.constant_names = [f"c{i+1}" for i in range(self.n_constants)]
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

        self.var_names = conditions.columns.tolist()
        self.TERMINALS = self.constant_names + self.var_names
        self.hall_of_fame = []
        self._expr_cache.clear()
        self.last_topk = []

        biomes = self._init_biomes()
        best_overall = (None, float("inf"), {})
        self.generation_log = []
        rounds_without_improvement = 0

        for gen in range(self.n_generation):
            if self._check_time() or self._stop_requested:
                if self.verbose:
                    print("Stop requested / time limit reached — finishing current fit.")
                break

            if self.verbose:
                print(f"\n=== Generation {gen+1}/{self.n_generation} ===")

            sample_size = min(500, len(conditions))
            idx = np.random.choice(len(conditions), size=sample_size, replace=False)
            Xs, ys = conditions.iloc[idx], observations.iloc[idx]

            means = []
            for b_i, b in enumerate(biomes):
                m = self._evolve_biome_one_gen(b, Xs, ys)
                means.append(m)
                if self.verbose:
                    print(f"[Biome {b_i}] mean fitness: {m:.5f} | best: {b.best[1]:.5f}")
                if self._check_time() or self._stop_requested:
                    break

            # Migration
            if (not (self._stop_requested or self._check_time())) and self.migration_interval and (gen + 1) % self.migration_interval == 0:
                if self.verbose:
                    print("... migration ...")
                self._migrate(biomes)

            # Global best
            prev_best = best_overall[1]
            for b in biomes:
                if b.best[1] < best_overall[1]:
                    best_overall = b.best

            self.generation_log.append(np.nanmean(means))

            if best_overall[1] < prev_best:
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if (not (self._stop_requested or self._check_time())) and self.early_stopping_rounds and rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose:
                        print("Early stopping: no improvement.")
                    break

        self.best_tree, self.best_fitness, self.best_c = best_overall
        self.best_equation = self._tree_translate(self.best_tree) if self.best_tree else None

        # Top-5 Snapshot (nur Anzeige, nicht für predict)
        self.hall_of_fame.sort(key=lambda x: x[1])
        self.last_topk = self.hall_of_fame[:5]

        if self.verbose and self.best_tree is not None:
            print("\nBest Equation:", self._pretty_equation(self.best_tree, self.best_c))
            print("Best MSE:", self.best_fitness)
            if self.last_topk:
                print("\nTop 5 candidates (not used for predict):")
                for i, (t, mse, consts) in enumerate(self.last_topk, 1):
                    print(f"  {i}. MSE={mse:.6g} | {self._pretty_equation(t, consts)}")

        return self

    def predict(self, conditions):
        if self.best_tree is None:
            raise ValueError("Model not fitted.")
        if isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=self.var_names)
        else:
            self.var_names = conditions.columns.tolist()

        # vektorisiert vorhersagen
        eq_str = self._tree_translate(self.best_tree)
        eq_template = self._prepare_equation_template(eq_str)
        code = self._compile_or_get(eq_template)
        preds = self._eval_expr_vectorized(code, conditions, self.best_c).reshape(-1, 1)

        if np.any(~np.isfinite(preds)):
            raise ValueError("Invalid prediction values.")
        return preds

    def print_eqn(self):
        if self.best_tree:
            pretty = self._pretty_equation(self.best_tree, self.best_c)
            print(f"Best equation: {pretty}")
            return pretty
        print("Model not fitted.")
        return ""
