# Full best version of NutsTheorists with all requested optimizations

from typing import List
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from itertools import product

warnings.filterwarnings('ignore', category=RuntimeWarning)

class NutsTheorists(BaseEstimator):
    def __init__(self, population_size=500, n_generation=20, mutation_rate=0.2, tournament_size=15,
                 early_stopping_rounds=5, complexity_penalty=0.02, n_constants=1, verbose=True, elitism=2):
        self.population_size = population_size
        self.n_generation = n_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.early_stopping_rounds = early_stopping_rounds
        self.complexity_penalty = complexity_penalty
        self.n_constants = n_constants
        self.elitism = elitism
        self.verbose = verbose

        self.UNARY_OPS = ['np.log', 'np.exp']
        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']
        self.hall_of_fame = []

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

    def _evaluate_tree_mse(self, tree, conditions, observations):
        key = str(tree)
        if self._is_constant_only(tree):
            if key in self._seen_constant_only:
                return tree, float("inf"), {k: 1.0 for k in self.constant_names}
            else:
                self._seen_constant_only.add(key)

        eq_str = self._tree_translate(tree)
        symbol_count = self._count_symbols(tree)
        if symbol_count > 40:
            return tree, float("inf"), {k: 1.0 for k in self.constant_names}

        def try_constants(constants):
            try:
                f = self._translate_tree_to_callable(eq_str, constants)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    preds = np.array([f(*row) for row in conditions.values])
                if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                    raise ValueError
                return preds
            except:
                raise ValueError

        best_constants, best_mse = None, float("inf")
        grid = np.linspace(0.1, 3.0, 5)

        for values in product(grid, repeat=len(self.constant_names)):
            constants = dict(zip(self.constant_names, values))
            try:
                preds = try_constants(constants)
                mse = mean_squared_error(observations.values.ravel(), preds.ravel())
                mse += self.complexity_penalty * symbol_count
                if mse < best_mse:
                    best_mse = mse
                    best_constants = constants
                if mse > 1000:
                    break
            except:
                continue

        mse = best_mse if best_constants else float("inf")
        best_constants = best_constants or {k: 1.0 for k in self.constant_names}
        if self.verbose:
            print(f"Evaluated: {eq_str}, MSE: {mse:.5f}, constants: {best_constants}, complexity: {symbol_count}")
        if np.isfinite(mse):
            self.hall_of_fame.append((copy.deepcopy(tree), mse, best_constants))
            self.hall_of_fame.sort(key=lambda x: x[1])
            self.hall_of_fame = self.hall_of_fame[:10]
        return tree, mse, best_constants

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

    def _tournament(self, population, k=50):
        selected = random.sample(population, min(len(population), self.tournament_size * k))
        return sorted(selected, key=lambda x: x[1])[:k]

    def generate_next_generation(self, top_k, max_depth=3):
        if len(top_k) < 2:
            return copy.deepcopy(top_k)
        new_pop = copy.deepcopy(top_k[:self.elitism])
        while len(new_pop) < self.population_size:
            p1, p2 = random.sample(top_k, 2)
            for child in self._crossover(p1, p2):
                mutated = self._mutate(child, max_depth)
                if self._count_symbols(mutated) <= 40:
                    try:
                        self._translate_tree_to_callable(self._tree_translate(mutated), self.best_c)
                        new_pop.append(mutated)
                    except:
                        continue
        return new_pop

    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        self.best_tree = None
        self.best_equation = None
        self.constant_names = [f"c{i+1}" for i in range(self.n_constants)] 
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

        self.var_names = conditions.columns.tolist()
        self.TERMINALS = self.constant_names + self.var_names
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

        population = [self._create_random_tree(5) for _ in range(self.population_size)]
        best_overall_mse = float('inf')
        rounds_without_improvement = 0
        self.generation_log = []
        self.best_mse_log = []
        self.best_eqn_log = []

        try:
            for gen in range(self.n_generation):
                if self.verbose:
                    print(f"\nGeneration {gen+1}:")
                sample_size = min(500, len(conditions))
                sample_idx = np.random.choice(len(conditions), size=sample_size, replace=False)
                X_sample = conditions.iloc[sample_idx]
                y_sample = observations.iloc[sample_idx]
                pop_scores = [self._evaluate_tree_mse(tree, X_sample, y_sample) for tree in population]
                top_k = self._tournament(pop_scores)
                top_k_trees = [tree for tree, _, _ in top_k]
                if len(top_k_trees) < 2:
                    print("Not enough individuals for crossover. Stopping early.")
                    break
                population = self.generate_next_generation(top_k_trees)

                valid_mses = [score for _, score, _ in pop_scores if np.isfinite(score)]
                best_tree, best_mse, best_c = min(pop_scores, key=lambda x: x[1])
                best_eq = self._tree_translate(best_tree)

                self.generation_log.append(np.mean(valid_mses) if valid_mses else float("inf"))
                self.best_mse_log.append(best_mse)
                self.best_eqn_log.append(best_eq)

                if best_mse < best_overall_mse:
                    best_overall_mse = best_mse
                    self.best_tree, self.best_c = best_tree, best_c
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if self.early_stopping_rounds and rounds_without_improvement >= self.early_stopping_rounds:
                    break

        except KeyboardInterrupt:
            print("\n[Interrupted by user] Stopping early and returning best result so far...")

        # final best
        if self.hall_of_fame:
            self.best_tree, self.best_fitness, self.best_c = min(self.hall_of_fame, key=lambda x: x[1])
            self.best_equation = self._tree_translate(self.best_tree)

        if self.verbose:
            print("\nBest Equation:", self.best_equation)
            print("Best MSE:", self.best_fitness)

        return self

    def predict(self, conditions):
        if self.best_tree is None:
            raise ValueError("Model not fitted.")
        if isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=self.var_names)
        else:
            self.var_names = conditions.columns.tolist()  # ensure correct mapping

        func = self._translate_tree_to_callable(self._tree_translate(self.best_tree), self.best_c)
        preds = np.array([func(*row) for row in conditions.values]).reshape(-1, 1)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            raise ValueError("Invalid prediction values.")
        return preds

    def print_eqn(self):
        if self.best_tree:
            eq = self._tree_translate(self.best_tree)
            print(f"Best equation: {eq}")
            return eq
        print("Model not fitted.")
        return ""

