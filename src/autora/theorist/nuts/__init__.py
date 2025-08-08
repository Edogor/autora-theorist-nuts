"""
Fixed Genetic Algorithm Implementation
"""
from typing import Union
import random
import copy
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error


class NutsTheorists(BaseEstimator):
    """
    Fixed genetic algorithm for symbolic regression
    """

    def __init__(self, population_size=100, n_generation=50, mutation_rate=0.15, tournament_size=3):
        self.population_size = population_size
        self.n_generation = n_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = 2  # Number of best trees to carry over unchanged

        # Attributes to store the final result
        self.best_equation = None
        self.best_equation_tree = None  # Store the actual tree structure
        self.best_params = None
        self.best_fitness = float('inf')  # Changed to inf since we minimize MSE
        self.result_list = []
        self.var_names = []  # Initialize empty, will be set in fit()
        
        # Operators
        self.UNARY_OPS = ['np.log', 'np.exp']
        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']
        self.BASE_TERMINALS = ['c']  # Keep constants separate
        self.TERMINALS = ['c']  # Will be updated in fit()

    def _create_random_tree(self, max_depth=3):
        """
        Recursively generates a single random equation tree.
        """
        if max_depth <= 0 or random.random() < 0.3:
            return random.choice(self.TERMINALS)

        # Recursive Step: Grow the tree
        chosen_op = random.choice(self.UNARY_OPS + self.BINARY_OPS)

        if chosen_op in self.UNARY_OPS:
            child = self._create_random_tree(max_depth - 1)
            return [chosen_op, child]
        
        elif chosen_op in self.BINARY_OPS:
            left_child = self._create_random_tree(max_depth - 1)
            right_child = self._create_random_tree(max_depth - 1)
            return [chosen_op, left_child, right_child]

    def _tree_translate(self, tree):
        """
        Recursively converts a nested equation tree into a string.
        """
        if isinstance(tree, str):  # base case: terminal symbol
            return tree

        op = tree[0]

        if op in ['+', '-', '*', '/']:
            left = self._tree_translate(tree[1])
            right = self._tree_translate(tree[2])
            return f"({left} {op} {right})"

        elif op == 'np.exp':
            arg = self._tree_translate(tree[1])
            return f"exp({arg})"

        elif op == 'np.log':
            arg = self._tree_translate(tree[1])
            return f"log({arg})"

        elif op == 'np.power':
            base = self._tree_translate(tree[1])
            exponent = self._tree_translate(tree[2])
            return f"({base} ^ {exponent})"

        else:
            raise ValueError(f"Unknown operator: {op}")
        
    def _translate_tree_to_callable(self, eq_str, var_names, constant_value=1.0):
        """
        Convert equation string to callable function.
        """
        eq_str = eq_str.replace('^', '**')
        eq_str = eq_str.replace('exp', 'np.exp').replace('log', 'np.log')

        if 'c' in eq_str:
            eq_str = eq_str.replace('c', str(constant_value))

        lambda_str = f"lambda {', '.join(var_names)}: {eq_str}"
        return eval(lambda_str, {"np": np})

    def _evaluate_tree_mse(self, tree, conditions, observations, constant_value=1.0):
        """
        Evaluates a tree and computes MSE.
        """
        try:
            eq_str = self._tree_translate(tree)
            func = self._translate_tree_to_callable(eq_str, conditions.columns.tolist(), constant_value)

            preds = np.array([func(*row) for row in conditions.values]).reshape(-1, 1)
            targets = np.array(observations).reshape(-1, 1)
            mse = mean_squared_error(targets, preds)

        except Exception:
            mse = float("inf")  # Penalize invalid equations

        return tree, mse

    def _tournament_selection(self, population_with_scores):
        """
        Performs a single tournament selection.
        Randomly selects tournament_size individuals and returns the best one.
        """
        tournament = random.sample(population_with_scores, min(self.tournament_size, len(population_with_scores)))
        # Return the best individual (lowest MSE)
        return min(tournament, key=lambda x: x[1])

    def _select_parents(self, population_with_scores, n_parents):
        """
        Select n_parents using tournament selection.
        """
        parents = []
        for _ in range(n_parents):
            winner = self._tournament_selection(population_with_scores)
            parents.append(winner[0])  # Just append the tree, not the score
        return parents

    def _crossover(self, tree1, tree2):
        """
        Perform crossover between two trees.
        """
        def get_random_subtree(tree):
            """
            Returns a randomly selected subtree with its parent and index.
            """
            if not isinstance(tree, list):
                return tree, None, None
            
            candidates = []
            
            def collect_subtrees(node, parent=None, idx=None):
                if isinstance(node, list):
                    candidates.append((node, parent, idx))
                    for i, child in enumerate(node[1:], 1):
                        collect_subtrees(child, node, i)
            
            collect_subtrees(tree)
            
            if not candidates:
                return tree, None, None
            
            return random.choice(candidates)

        # Deep copy to avoid modifying originals
        t1 = copy.deepcopy(tree1)
        t2 = copy.deepcopy(tree2)

        # Get random subtrees
        node1, parent1, idx1 = get_random_subtree(t1)
        node2, parent2, idx2 = get_random_subtree(t2)

        # Perform crossover if both have valid parents
        if parent1 is not None and parent2 is not None:
            # Swap subtrees
            parent1[idx1] = copy.deepcopy(node2)
            parent2[idx2] = copy.deepcopy(node1)

        return t1, t2

    def _mutate(self, tree, max_depth=3):
        """
        Mutate a tree with proper depth control.
        """
        def recursive_mutate(node, depth=0):
            if not isinstance(node, list):
                # Terminal mutation
                if random.random() < self.mutation_rate:
                    return random.choice(self.TERMINALS)
                return node

            # Operator or subtree mutation
            if random.random() < self.mutation_rate:
                # FIX: Ensure we never pass negative max_depth
                remaining_depth = max(1, max_depth - depth)  # Always at least 1
                return self._create_random_tree(remaining_depth)
            
            # Recursively mutate children
            mutated = [node[0]]
            for child in node[1:]:
                mutated.append(recursive_mutate(child, depth + 1))
            
            return mutated

        return recursive_mutate(copy.deepcopy(tree))

    def generate_next_generation(self, population_with_scores):
        """
        Generate the next generation using genetic operators.
        """
        new_population = []
        
        # Step 1: Elitism - keep the best individuals unchanged
        sorted_pop = sorted(population_with_scores, key=lambda x: x[1])
        for i in range(min(self.elitism, len(sorted_pop))):
            new_population.append(copy.deepcopy(sorted_pop[i][0]))
        
        # Step 2: Fill the rest with crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            if random.random() < 0.8:  # 80% crossover rate
                parent1 = self._tournament_selection(population_with_scores)[0]
                parent2 = self._tournament_selection(population_with_scores)[0]
                
                # Perform crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Apply mutation
                child1 = self._mutate(child1)
                if len(new_population) < self.population_size - 1:
                    child2 = self._mutate(child2)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(child1)
            else:
                # Direct reproduction with mutation
                parent = self._tournament_selection(population_with_scores)[0]
                child = self._mutate(copy.deepcopy(parent))
                new_population.append(child)
        
        # Ensure we have exactly population_size individuals
        return new_population[:self.population_size]

    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """
        Runs the genetic algorithm to find the best equation.
        """
        # 1. INITIALIZATION
        if isinstance(conditions, pd.DataFrame):
            conditions = conditions.copy()
        elif isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=[f'x{i+1}' for i in range(conditions.shape[1])])
        
        # Set up variable names and terminals
        self.var_names = conditions.columns.tolist()
        self.TERMINALS = self.BASE_TERMINALS + self.var_names  # Reset and add var names
        
        y_true = observations.values.ravel()

        # Create initial population with varied depths
        population = []
        for _ in range(self.population_size):
            depth = random.randint(2, 4)  # Varied initial depths
            population.append(self._create_random_tree(depth))
        
        # Track best ever solution
        self.best_fitness = float('inf')
        self.best_equation_tree = None
        
        # 2. MAIN EVOLUTION LOOP
        for generation in range(self.n_generation):
            # Evaluate current population
            pop_with_scores = []
            for tree in population:
                tree, fitness = self._evaluate_tree_mse(tree, conditions, observations, constant_value=1.0)
                pop_with_scores.append((tree, fitness))
            
            # Track the best solution
            current_best = min(pop_with_scores, key=lambda x: x[1])
            if current_best[1] < self.best_fitness:
                self.best_fitness = current_best[1]
                self.best_equation_tree = copy.deepcopy(current_best[0])
                print(f"Gen {generation+1}: New best fitness (MSE) = {self.best_fitness:.6f}")
            
            # Generate next generation
            population = self.generate_next_generation(pop_with_scores)
        
        # Store final best equation
        if self.best_equation_tree is not None:
            self.best_equation = self._tree_translate(self.best_equation_tree)
        else:
            # Fallback to best of final generation
            final_best = min(pop_with_scores, key=lambda x: x[1])
            self.best_equation_tree = final_best[0]
            self.best_equation = self._tree_translate(final_best[0])
            self.best_fitness = final_best[1]
        
        print("\nEvolution finished.")
        print(f"Best equation found: {self.best_equation}")
        print(f"Best fitness (MSE): {self.best_fitness:.6f}")
        
        return self

    def predict(self, conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Make predictions using the best equation found.
        """
        if self.best_equation_tree is None:
            raise ValueError("No equation available. Did you forget to call fit()?")

        if isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=self.var_names)

        eq_str = self._tree_translate(self.best_equation_tree)
        func = self._translate_tree_to_callable(eq_str, self.var_names, constant_value=1.0)

        preds = np.array([func(*row) for row in conditions.values]).reshape(-1, 1)
        return preds
    
    def print_eqn(self):
        """
        Print the best equation found.
        """
        if self.best_equation is None:
            print("No equation available. Did you forget to call fit()?")
            return
        
        print(f"Best equation: {self.best_equation}")
        return self.best_equation