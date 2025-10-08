# Full best version of NutsTheorists with all requested optimizations

import copy
import random
import warnings
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Island:
    """
    Represents an island in the island model genetic algorithm.
    Each island has its own population and can have different parameters and heuristics.
    """

    def __init__(
        self,
        name: str,
        population_size: int = 100,
        mutation_rate: float = 0.2,
        tournament_size: int = 20,
        elitism: int = 10,
        operator_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize an island with specific parameters.

        Args:
            name: Unique identifier for the island
            population_size: Number of individuals in the island population
            mutation_rate: Probability of mutation for each individual
            tournament_size: Size of tournament selection
            elitism: Number of elite individuals to preserve
            operator_weights: Dictionary mapping operator types to preference weights.
                             Keys: 'unary', 'binary', 'crossover', 'mutation'
                             Values: relative weights (will be normalized)
        """
        self.name = name
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        # Default equal weights if not specified
        if operator_weights is None:
            operator_weights = {"unary": 1.0, "binary": 1.0, "crossover": 1.0, "mutation": 1.0}
        self.operator_weights = operator_weights

        # Island state
        self.population = []
        self.best_tree = None
        self.best_fitness = float("inf")
        self.best_constants = {}
        self.generation_history = []
        self.is_alive = True

    def get_operator_weight(self, operator_type: str) -> float:
        """Get the weight for a specific operator type."""
        return self.operator_weights.get(operator_type, 1.0)

    def update_stats(self, generation: int, best_tree, best_fitness, best_constants):
        """Update island statistics after a generation."""
        if best_fitness < self.best_fitness:
            self.best_tree = copy.deepcopy(best_tree)
            self.best_fitness = best_fitness
            self.best_constants = copy.deepcopy(best_constants)

        self.generation_history.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "population_size": len(self.population),
            }
        )

    def __repr__(self):
        return (
            f"Island(name='{self.name}', population_size={self.population_size}, "
            f"mutation_rate={self.mutation_rate}, best_fitness={self.best_fitness:.6f}, "
            f"alive={self.is_alive})"
        )


class IslandModel:
    """
    Manages multiple islands for parallel evolution with different parameters.
    """

    def __init__(self, islands: List[Island]):
        """
        Initialize the island model.

        Args:
            islands: List of Island objects to manage
        """
        self.islands = islands
        self.global_best_tree = None
        self.global_best_fitness = float("inf")
        self.global_best_constants = {}
        self.global_best_island = None

    def get_best_island(self) -> Optional[Island]:
        """Return the island with the best fitness."""
        alive_islands = [isl for isl in self.islands if isl.is_alive]
        if not alive_islands:
            return None
        return min(alive_islands, key=lambda x: x.best_fitness)

    def update_global_best(self):
        """Update global best from all islands."""
        for island in self.islands:
            if island.is_alive and island.best_fitness < self.global_best_fitness:
                self.global_best_fitness = island.best_fitness
                self.global_best_tree = copy.deepcopy(island.best_tree)
                self.global_best_constants = copy.deepcopy(island.best_constants)
                self.global_best_island = island.name

    def get_island_rankings(self) -> List[Dict]:
        """
        Get rankings of all islands by performance.

        Returns:
            List of dictionaries with island statistics, sorted by fitness
        """
        rankings = []
        for island in self.islands:
            rankings.append(
                {
                    "name": island.name,
                    "best_fitness": island.best_fitness,
                    "population_size": island.population_size,
                    "mutation_rate": island.mutation_rate,
                    "alive": island.is_alive,
                    "generations_completed": len(island.generation_history),
                }
            )

        return sorted(rankings, key=lambda x: x["best_fitness"])

    def print_island_summary(self):
        """Print a summary of all islands."""
        print("\n" + "=" * 80)
        print("ISLAND MODEL SUMMARY")
        print("=" * 80)
        rankings = self.get_island_rankings()

        print(f"\nGlobal Best Island: {self.global_best_island}")
        print(f"Global Best Fitness: {self.global_best_fitness:.6f}")

        print("\nIsland Rankings:")
        header = (
            f"{'Rank':<6}{'Island':<20}{'Fitness':<15}"
            f"{'Pop Size':<12}{'Mut Rate':<12}{'Status':<10}"
        )
        print(header)
        print("-" * 80)

        for rank, stats in enumerate(rankings, 1):
            status = "ALIVE" if stats["alive"] else "DEAD"
            print(
                f"{rank:<6}{stats['name']:<20}{stats['best_fitness']:<15.6f}"
                f"{stats['population_size']:<12}{stats['mutation_rate']:<12.3f}{status:<10}"
            )

        print("=" * 80)

    def kill_weak_islands(self, fitness_threshold: float):
        """
        Mark islands as dead if their fitness exceeds the threshold.

        Args:
            fitness_threshold: Fitness value above which islands are killed
        """
        for island in self.islands:
            if island.is_alive and island.best_fitness > fitness_threshold:
                island.is_alive = False
                island.population = []

    def get_migration_candidates(self, n_islands: int = 2, n_trees: int = 5) -> List:
        """
        Get top trees from best islands for potential migration/crossbreeding.

        Args:
            n_islands: Number of top islands to select from
            n_trees: Number of top trees to take from each island

        Returns:
            List of (tree, fitness, constants) tuples
        """
        alive_islands = [isl for isl in self.islands if isl.is_alive]
        if len(alive_islands) < n_islands:
            n_islands = len(alive_islands)

        # Get top n_islands by fitness
        top_islands = sorted(alive_islands, key=lambda x: x.best_fitness)[:n_islands]

        candidates = []
        for island in top_islands:
            # Get top trees from each island's population
            if hasattr(island, "best_tree") and island.best_tree is not None:
                candidates.append(
                    (
                        copy.deepcopy(island.best_tree),
                        island.best_fitness,
                        copy.deepcopy(island.best_constants),
                    )
                )

        return candidates[: n_trees * n_islands]


class NutsTheorists(BaseEstimator):
    def __init__(
        self,
        population_size=500,
        n_generation=30,
        mutation_rate=0.2,
        tournament_size=20,
        early_stopping_rounds=5,
        complexity_penalty=0.02,
        n_constants=1,
        verbose=True,
        elitism=50,
        constant_grid=None,
        use_island_model=False,
        islands=None,
    ):
        """
        Initialize NutsTheorists genetic programming system.

        Args:
            population_size: Size of the population (if not using island model)
            n_generation: Number of generations to evolve
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            early_stopping_rounds: Number of rounds without improvement before stopping
            complexity_penalty: Penalty for equation complexity
            n_constants: Number of constants in equations
            verbose: Whether to print progress
            elitism: Number of elite individuals to preserve
            constant_grid: Grid for constant optimization
            use_island_model: Whether to use the island model
            islands: List of Island objects (if use_island_model=True)
        """
        self.population_size = population_size
        self.n_generation = n_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.early_stopping_rounds = early_stopping_rounds
        self.complexity_penalty = complexity_penalty
        self.n_constants = n_constants
        self.elitism = elitism
        self.verbose = verbose

        # New param: grid for constant tuning
        self.constant_grid = constant_grid or np.linspace(-3, 3.0, 20)

        self.UNARY_OPS = ["np.log", "np.exp"]
        self.BINARY_OPS = ["+", "-", "*", "/", "np.power"]
        self.hall_of_fame = []

        # Island model parameters
        self.use_island_model = use_island_model
        self.island_model = None
        if use_island_model and islands:
            self.island_model = IslandModel(islands)
        elif use_island_model and not islands:
            # Create default islands with different parameters if none provided
            self.island_model = IslandModel(self._create_default_islands())

    def _create_default_islands(self) -> List[Island]:
        """Create a set of default islands with varying parameters."""
        return [
            Island(
                "Conservative",
                population_size=100,
                mutation_rate=0.1,
                tournament_size=15,
                elitism=20,
                operator_weights={"unary": 1.5, "binary": 1.0, "crossover": 1.2, "mutation": 0.8},
            ),
            Island(
                "Balanced",
                population_size=100,
                mutation_rate=0.2,
                tournament_size=20,
                elitism=10,
                operator_weights={"unary": 1.0, "binary": 1.0, "crossover": 1.0, "mutation": 1.0},
            ),
            Island(
                "Aggressive",
                population_size=100,
                mutation_rate=0.4,
                tournament_size=25,
                elitism=5,
                operator_weights={"unary": 0.8, "binary": 1.5, "crossover": 0.8, "mutation": 1.5},
            ),
            Island(
                "Exploratory",
                population_size=100,
                mutation_rate=0.3,
                tournament_size=30,
                elitism=8,
                operator_weights={"unary": 1.2, "binary": 1.2, "crossover": 1.0, "mutation": 1.3},
            ),
        ]

    def get_island_summary(self) -> Optional[Dict]:
        """
        Get summary of island performance if using island model.

        Returns:
            Dictionary with island statistics or None if not using island model
        """
        if not self.use_island_model or not self.island_model:
            return None

        return {
            "rankings": self.island_model.get_island_rankings(),
            "global_best_island": self.island_model.global_best_island,
            "global_best_fitness": self.island_model.global_best_fitness,
        }

    def print_island_summary(self):
        """Print summary of island performance."""
        if self.use_island_model and self.island_model:
            self.island_model.print_island_summary()
        else:
            print("Island model not enabled.")

    def _init_constants_and_terminals(self, var_names):
        self.var_names = var_names
        self.constant_names = [f"c{i+1}" for i in range(self.n_constants)]
        self.TERMINALS = self.constant_names + self.var_names
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

    def _create_random_tree(self, max_depth=3, island: Optional[Island] = None):
        """
        Create a random tree, optionally respecting island operator preferences.

        Args:
            max_depth: Maximum depth of the tree
            island: Optional island object to use for operator weights
        """
        if max_depth == 0 or random.random() < 0.2:
            return random.choice(self.TERMINALS)

        # Get operator weights if island is specified
        if island:
            unary_weight = island.get_operator_weight("unary")
            binary_weight = island.get_operator_weight("binary")
        else:
            unary_weight = binary_weight = 1.0

        # Choose operator type based on weights
        total_weight = unary_weight + binary_weight
        if random.random() < unary_weight / total_weight:
            op = random.choice(self.UNARY_OPS)
            return [op, self._create_random_tree(max_depth - 1, island)]
        else:
            op = random.choice(self.BINARY_OPS)
            return [
                op,
                self._create_random_tree(max_depth - 1, island),
                self._create_random_tree(max_depth - 1, island),
            ]

    def _tree_translate(self, tree):
        if isinstance(tree, str):
            return tree
        op, *args = tree
        translated = [self._tree_translate(arg) for arg in args]
        if op == "np.exp":
            return f"exp({translated[0]})"
        elif op == "np.log":
            return f"log({translated[0]})"
        elif op == "np.power":
            return f"({translated[0]} ^ {translated[1]})"
        return f"({translated[0]} {op} {translated[1]})"

    def _prepare_equation(self, eq_str, constants):
        eq_str = eq_str.replace("^", "**").replace("exp", "np.exp").replace("log", "np.log")
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
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    preds = np.array([f(*row) for row in conditions.values])
                if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                    raise ValueError
                return preds
            except Exception:
                raise ValueError

        best_constants, best_mse = None, float("inf")
        grid = np.linspace(-5, 5.0, 40)

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
            except Exception:
                continue

        mse = best_mse if best_constants else float("inf")
        best_constants = best_constants or {k: 1.0 for k in self.constant_names}
        if self.verbose:
            msg = (
                f"Evaluated: {eq_str}, MSE: {mse:.5f}, "
                f"constants: {best_constants}, complexity: {symbol_count}"
            )
            print(msg)
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
        node, parent, idx, _ = random.choices(candidates, weights=[d + 1 for *_, d in candidates])[
            0
        ]
        return node, parent, idx

    def _crossover(self, t1, t2):
        t1, t2 = copy.deepcopy(t1), copy.deepcopy(t2)
        n1, p1, i1 = self._get_random_subtree(t1)
        n2, p2, i2 = self._get_random_subtree(t2)
        if p1 and p2:
            p1[i1], p2[i2] = n2, n1
        return t1, t2

    def _mutate(self, tree, max_depth=3, island: Optional[Island] = None):
        """
        Mutate a tree, optionally using island-specific mutation rate.

        Args:
            tree: Tree to mutate
            max_depth: Maximum depth for new subtrees
            island: Optional island object to use for mutation rate
        """
        mutation_rate = island.mutation_rate if island else self.mutation_rate

        def recurse(node):
            if not isinstance(node, list):
                return random.choice(self.TERMINALS) if random.random() < mutation_rate else node
            if random.random() < mutation_rate:
                return self._create_random_tree(max_depth, island)
            return [node[0]] + [recurse(child) for child in node[1:]]

        return recurse(copy.deepcopy(tree))

    def _tournament(self, population, k=50, island: Optional[Island] = None):
        """
        Tournament selection, optionally using island-specific tournament size.

        Args:
            population: List of (tree, fitness, constants) tuples
            k: Number of individuals to select
            island: Optional island object to use for tournament size
        """
        tournament_size = island.tournament_size if island else self.tournament_size
        selected = random.sample(population, min(len(population), tournament_size * k))
        return sorted(selected, key=lambda x: x[1])[:k]

    def generate_next_generation(self, top_k, max_depth=3, island: Optional[Island] = None):
        """
        Generate next generation, optionally using island-specific parameters.

        Args:
            top_k: Top k individuals from previous generation
            max_depth: Maximum tree depth
            island: Optional island object for island-specific parameters
        """
        if len(top_k) < 2:
            return copy.deepcopy(top_k)

        elitism = island.elitism if island else self.elitism
        pop_size = island.population_size if island else self.population_size

        new_pop = copy.deepcopy(top_k[:elitism])  # ← step 1: keep elite trees
        while len(new_pop) < pop_size:  # ← step 2: fill rest of population
            p1, p2 = random.sample(top_k, 2)  # ← pick 2 parents from top_k
            for child in self._crossover(p1, p2):  # ← create 2 children by swapping parts
                mutated = self._mutate(child, max_depth, island)  # ← randomly mutate the child
                if self._count_symbols(mutated) <= 40:  # ← enforce max length
                    try:
                        self._translate_tree_to_callable(
                            self._tree_translate(mutated), self.best_c
                        )
                        new_pop.append(mutated)
                    except Exception:
                        continue
        return new_pop

    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """
        Fit the model using either standard or island-based evolution.

        Args:
            conditions: Input features DataFrame
            observations: Target values DataFrame

        Returns:
            self
        """
        self.best_tree = None
        self.best_equation = None
        self.constant_names = [f"c{i+1}" for i in range(self.n_constants)]
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

        self.var_names = conditions.columns.tolist()
        self.TERMINALS = self.constant_names + self.var_names
        self.best_c = {name: 1.0 for name in self.constant_names}
        self._seen_constant_only = set()

        if self.use_island_model and self.island_model:
            return self._fit_with_islands(conditions, observations)
        else:
            return self._fit_standard(conditions, observations)

    def _fit_standard(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """Standard single-population evolution."""
        population = [self._create_random_tree(5) for _ in range(self.population_size)]
        best_overall_mse = float("inf")
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
                pop_scores = [
                    self._evaluate_tree_mse(tree, X_sample, y_sample) for tree in population
                ]
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

                if (
                    self.early_stopping_rounds
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    break

        except KeyboardInterrupt:
            print("\n[Interrupted by user] Stopping early and returning best result so far...")

        # final best
        if self.hall_of_fame:
            self.best_tree, self.best_fitness, self.best_c = min(
                self.hall_of_fame, key=lambda x: x[1]
            )
            self.best_equation = self._tree_translate(self.best_tree)

        if self.verbose:
            print("\nBest Equation:", self.best_equation)
            print("Best MSE:", self.best_fitness)

        return self

    def _fit_with_islands(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """Island model evolution with multiple independent populations."""
        # Initialize each island's population
        for island in self.island_model.islands:
            island.population = [
                self._create_random_tree(5, island) for _ in range(island.population_size)
            ]

        self.generation_log = []
        self.best_mse_log = []
        self.best_eqn_log = []

        try:
            for gen in range(self.n_generation):
                if self.verbose:
                    print(f"\nGeneration {gen+1}:")

                # Sample data once for all islands
                sample_size = min(500, len(conditions))
                sample_idx = np.random.choice(len(conditions), size=sample_size, replace=False)
                X_sample = conditions.iloc[sample_idx]
                y_sample = observations.iloc[sample_idx]

                # Evolve each island independently
                for island in self.island_model.islands:
                    if not island.is_alive:
                        continue

                    # Evaluate population
                    pop_scores = [
                        self._evaluate_tree_mse(tree, X_sample, y_sample)
                        for tree in island.population
                    ]

                    # Tournament selection
                    top_k = self._tournament(pop_scores, k=50, island=island)
                    top_k_trees = [tree for tree, _, _ in top_k]

                    if len(top_k_trees) < 2:
                        if self.verbose:
                            print(
                                f"Island '{island.name}': Not enough individuals. Marking as dead."
                            )
                        island.is_alive = False
                        continue

                    # Generate next generation
                    island.population = self.generate_next_generation(top_k_trees, island=island)

                    # Update island statistics
                    best_tree, best_mse, best_c = min(pop_scores, key=lambda x: x[1])
                    island.update_stats(gen, best_tree, best_mse, best_c)

                    if self.verbose:
                        print(f"Island '{island.name}': Best MSE = {best_mse:.6f}")

                # Update global best from all islands
                self.island_model.update_global_best()

                # Track overall progress
                self.generation_log.append(gen)
                self.best_mse_log.append(self.island_model.global_best_fitness)
                if self.island_model.global_best_tree:
                    self.best_eqn_log.append(
                        self._tree_translate(self.island_model.global_best_tree)
                    )

        except KeyboardInterrupt:
            print("\n[Interrupted by user] Stopping early and returning best result so far...")

        # Set final best from island model
        self.island_model.update_global_best()
        if self.island_model.global_best_tree:
            self.best_tree = self.island_model.global_best_tree
            self.best_fitness = self.island_model.global_best_fitness
            self.best_c = self.island_model.global_best_constants
            self.best_equation = self._tree_translate(self.best_tree)

        if self.verbose:
            print("\n" + "=" * 80)
            print("EVOLUTION COMPLETE")
            print("=" * 80)
            self.island_model.print_island_summary()
            print("\nGlobal Best Equation:", self.best_equation)
            print("Global Best MSE:", self.best_fitness)

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
            raw_eq = self._tree_translate(self.best_tree)
            prepared_eq = self._prepare_equation(raw_eq, self.best_c)
            print(f"Best equation: {prepared_eq}")
            return prepared_eq
        print("Model not fitted.")
        return ""
