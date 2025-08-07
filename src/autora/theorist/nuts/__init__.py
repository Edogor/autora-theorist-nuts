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
    A Genetic Algorithm-based theorist to discover symbolic equations.
    This is a corrected and fully integrated version.
    """

    def __init__(self, population_size=50, n_generations=30, mutation_rate=0.15, tournament_size=3, elitism=2, max_depth=4):
        # --- GA Hyperparameters ---
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.max_depth = max_depth

        # --- Internal Attributes (will be set during .fit()) ---
        self.best_equation_ = None
        self.best_params_ = None
        self.best_fitness_ = -float('inf')
        
        self.UNARY_OPS = ['np.log', 'np.exp']
        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']
        self.TERMINALS = [] # Will be set dynamically in .fit()
        self.variable_names_ = []

    # --- 1. HELPER METHODS FOR TREE MANIPULATION ---

    def _create_random_tree(self, max_depth):
        """Recursively generates a single random equation tree."""
        if max_depth == 0 or random.random() < 0.3:
            return random.choice(self.TERMINALS)

        chosen_op = random.choice(self.UNARY_OPS + self.BINARY_OPS)

        if chosen_op in self.UNARY_OPS:
            child = self._create_random_tree(max_depth - 1)
            return [chosen_op, child]
        else: # Binary op
            left = self._create_random_tree(max_depth - 1)
            right = self._create_random_tree(max_depth - 1)
            return [chosen_op, left, right]

    def _tree_to_lambda_string(self, tree):
        """Recursively converts a tree to an evaluatable string."""
        if not isinstance(tree, list):
            return str(tree)
        
        op = tree[0]
        if op == 'np.power':
            base = self._tree_to_lambda_string(tree[1])
            exponent = self._tree_to_lambda_string(tree[2])
            return f"np.power({base}, {exponent})"

        elif op in self.UNARY_OPS:
            child_str = self._tree_to_lambda_string(tree[1])
            return f"{op}({child_str})"
            
        elif op in self.BINARY_OPS: # Handles '+', '-', '*', '/'
            left_str = self._tree_to_lambda_string(tree[1])
            right_str = self._tree_to_lambda_string(tree[2])
            return f"({left_str} {op} {right_str})"

    # --- 2. FITNESS EVALUATION (CRITICAL FIX) ---

    def _get_fitness(self, tree, X_df, y_series):
        """
        Calculates fitness by using curve_fit to find the best constant 'c'
        and then computing the negative MSE.
        """
        try:
            # a. Convert the tree into a callable function string
            func_str = self._tree_to_lambda_string(tree)
            
            # b. Create the target function for curve_fit. 
            # It must take X data and the parameters to be fitted (just 'c').
            # The lambda function will unpack the dataframe columns.
            var_map = {name: f"X['{name}']" for name in self.variable_names_}
            lambda_body = func_str
            for var, access_str in var_map.items():
                lambda_body = lambda_body.replace(var, access_str)

            target_func = eval(f"lambda X, c: {lambda_body}", {"np": np})

            # c. Use curve_fit to find the optimal value for the constant 'c'.
            # This is the most important step for accurate evaluation.
            params, _ = curve_fit(target_func, X_df, y_series, p0=[1.0], maxfev=5000)
            best_c = params[0]

            # d. Make predictions with the best-fit constant and calculate MSE.
            predictions = target_func(X_df, best_c)
            mse = mean_squared_error(y_series, predictions)

            # e. Return negative MSE as fitness (higher is better).
            return -mse, params

        except (RuntimeError, TypeError, ValueError, ZeroDivisionError):
            # If curve_fit fails or the equation is invalid, assign worst fitness.
            return -float('inf'), None

    # --- 3. GENETIC OPERATORS ---

    def _tournament_selection(self, population_with_scores):
        """Selects one parent using a tournament."""
        tournament = random.sample(population_with_scores, self.tournament_size)
        winner = max(tournament, key=lambda item: item[1])
        return winner[0] # Return the tree

    def _crossover(self, parent1, parent2):
        """Performs crossover by swapping random sub-trees."""
        child = copy.deepcopy(parent1)
        # Simplified crossover: replace a random part of parent1 with a random part of parent2
        if isinstance(child, list) and len(child) > 1:
            idx_to_replace = random.randint(1, len(child) - 1)
            if isinstance(parent2, list) and len(parent2) > 1:
                replacement_node = copy.deepcopy(random.choice(parent2[1:]))
                child[idx_to_replace] = replacement_node
        return child

    def _mutate(self, tree):
        """Applies mutation by replacing a random node with a new random sub-tree."""
        if random.random() < self.mutation_rate:
            if isinstance(tree, list) and len(tree) > 1:
                idx_to_mutate = random.randint(1, len(tree) - 1)
                tree[idx_to_mutate] = self._create_random_tree(max_depth=1)
        return tree

    # --- 4. MAIN FIT AND PREDICT METHODS ---

    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """Runs the genetic algorithm to find the best equation."""
        # --- Initialization ---
        self.variable_names_ = list(conditions.columns)
        self.TERMINALS = self.variable_names_ + ['c']
        y_true = observations.values.ravel()
        population = [self._create_random_tree(self.max_depth) for _ in range(self.population_size)]

        # --- Main Evolution Loop ---
        for generation in range(self.n_generations):
            # a. Evaluate population
            pop_with_scores = []
            for tree in population:
                fitness, params = self._get_fitness(tree, conditions, y_true)
                pop_with_scores.append((tree, fitness, params))

            # b. Sort and track the best individual
            pop_with_scores.sort(key=lambda item: item[1], reverse=True)
            if pop_with_scores[0][1] > self.best_fitness_:
                self.best_equation_, self.best_fitness_, self.best_params_ = pop_with_scores[0]
                print(f"Gen {generation+1}: New best fitness = {self.best_fitness_:.4f}")

            # c. Create the next generation
            new_population = []
            elites = [tree for tree, _, _ in pop_with_scores[:self.elitism]]
            new_population.extend(elites)
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(pop_with_scores)
                parent2 = self._tournament_selection(pop_with_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population

        print("\nEvolution finished.")
        return self

    def predict(self, conditions: pd.DataFrame):
        """Makes predictions using the best-found equation."""
        if self.best_equation_ is None:
            raise RuntimeError("You must call .fit() before .predict()")

        func_str = self._tree_to_lambda_string(self.best_equation_)
        var_map = {name: f"X['{name}']" for name in self.variable_names_}
        lambda_body = func_str
        for var, access_str in var_map.items():
            lambda_body = lambda_body.replace(var, access_str)
        
        best_c = self.best_params_[0] if self.best_params_ is not None else 1.0
        
        final_func = eval(f"lambda X, c: {lambda_body}", {"np": np})
        predictions = final_func(conditions, best_c)
        return predictions.values.reshape(-1, 1)

    def print_eqn(self):
        """Returns the best-found equation as a string."""
        if self.best_equation_ is None:
            return "Model has not been fitted yet."
        
        # This is a simplified string translation. You can make it prettier.
        return self._tree_to_lambda_string(self.best_equation_)

