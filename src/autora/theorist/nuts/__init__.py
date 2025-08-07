"""
Example Theorist
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
    """

    def __init__(self, population_size=100, n_generations=50, mutation_rate=0.1, tournament_size=3):
        self.population_size = population_size
        self.n_generation = n_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        #Attributes to store the final result
        self.best_equation = None
        self.best_params = None
        self.best_fitness = -1
        self.UNARY_OPS = ['np.log', 'np.exp']
        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']
        self.TERMINALS = ['S1', 'S2', 'c']

    def _create_random_tree(self, max_depth = 3):
        """
        Recursively generates a single random equation tree.

        The tree is represented as a nested list.
        E.g., ['+', 'S1', ['*', 'c', 'S2']] represents S1 + (c * S2)

        Args:
            max_depth (int): The maximum depth of the tree to generate. This
                             prevents infinitely long equations.

        Returns:
            list: A nested list representing the equation tree.
        """
        # --- Base Case: Stop growing the tree ---
        # If we reach max depth, we MUST choose a terminal to end the branch.
        # We also add a small chance to pick a terminal even if we're not at
        # max depth, which allows for trees of varying shapes and sizes.
        if max_depth == 0 or random.random() < 0.2:
            return random.choice(self.TERMINALS)

        # --- Recursive Step: Grow the tree ---
        # Choose an operator from all available operators.
        chosen_op = random.choice(self.UNARY_OPS + self.BINARY_OPS)

        # Build the branch based on the type of operator.
        if chosen_op in self.UNARY_OPS:
            # A unary operator has one child.
            # We call the function again to create that child, reducing the depth.
            child = self._create_random_tree(max_depth - 1)
            return [chosen_op, child]
        
        elif chosen_op in self.BINARY_OPS:
            # A binary operator has two children.
            # We call the function twice to create the left and right children.
            left_child = self._create_random_tree(max_depth - 1)
            right_child = self._create_random_tree(max_depth - 1)
            return [chosen_op, left_child, right_child]

    def _tree_translate(self, tree):
        """
        Recursively converts a nested equation tree into a string.

        Args:
            tree (list or str): The equation tree.

        Returns:
            str: String representation of the equation.
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
        

    def generate_next_generation(top_k_trees, pop_size=1000, mutation_rate=0.2, max_depth=3, elitism=2):
        print("next generation")
        """
        Generate the next generation from top-performing trees.
        
        Args:
            top_k_trees (list): Top-performing trees (equation trees).
            pop_size (int): Total population size to generate.
            mutation_rate (float): Probability of mutating a node.
            max_depth (int): Max depth for new subtrees during mutation.
            elitism (int): Number of best trees to carry over unchanged.
    
        Returns:
            list: New generation of equation trees.
        """
    
        def crossover(tree1, tree2):
            print("Crossover between trees:")
            print(tree1)
            print(tree2)
            def get_random_subtree(tree):
                if not isinstance(tree, list):
                    return tree, None, None
                idx = random.randint(1, len(tree)-1)
                return tree[idx], tree, idx
    
            t1 = copy.deepcopy(tree1)
            t2 = copy.deepcopy(tree2)
    
            node1, parent1, idx1 = get_random_subtree(t1)
            node2, parent2, idx2 = get_random_subtree(t2)
    
            if parent1 is not None and parent2 is not None:
                parent1[idx1], parent2[idx2] = node2, node1
    
            return t1, t2

        def mutate(tree):
            print("Mutating tree:")
            def recursive_mutate(node, depth=0):
                if not isinstance(node, list):
                    # Terminal mutation
                    if random.random() < mutation_rate:
                        return random.choice(['S1', 'S2', 'c'])
                    return node
    
                # Subtree mutation
                if random.random() < mutation_rate:
                    return NutsTheorists()._create_random_tree(max_depth)
    
                # Recurse through children
                return [node[0]] + [recursive_mutate(child, depth+1) for child in node[1:]]
    
            return recursive_mutate(copy.deepcopy(tree))
    
        new_population = []

        # Step 1: Elitism — carry over best performers unchanged
        new_population.extend(copy.deepcopy(top_k_trees[:elitism]))
    
        # Step 2: Crossover and mutation
        while len(new_population) < pop_size:
            p1, p2 = random.sample(top_k_trees, 2)
            child1, child2 = crossover(p1, p2)
            new_population.append(mutate(child1))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2))
    
        return new_population
      
        
    def tree_to_function(tree):
        """
        Converts a nested list expression like ['+', 'S1', ['*', 'S2', 'c']]
        into a Python function that can be evaluated with input data.
        """
#hey

        def _convert(node):
            if isinstance(node, list):
                if len(node) == 2:  # Unary op
                    return f"{node[0]}({_convert(node[1])})"
                elif len(node) == 3:  # Binary op
                    return f"({_convert(node[1])} {node[0]} {_convert(node[2])})"
            else:
                return str(node)

        expr = _convert(tree)

        return lambda cond: eval(expr, {
            "np": np,
            "S1": cond[:, 0],
            "S2": cond[:, 1],
            "c": 1.0
        })

    
    def fitness_function(expression_func, conditions, observations):
        try:
            preds = expression_func(conditions)
            return -mean_squared_error(observations, preds)
        except Exception:
            return -float("inf") 

   def append_tree_score(tree, mse):
        if not hasattr(append_tree_score, "result_list"):
            append_tree_score.result_list = []  # initialize list once

        append_tree_score.result_list.append((tree, mse))
        return append_tree_score.result_list


    def _tournament(self, population_with_scores):
        """
        Selects a single parent from the population using tournament selection.

        Args:
            population_with_scores (list): A list of tuples, where each tuple is
                                           (tree, mse_score).

        Returns:
            list: The tree of the winning individual, who will be a parent.
        """
        # 1. Randomly select individuals for the tournament.
        tournament_entrants = random.sample(population_with_scores, self.tournament_size)
        
        # 2. Find the winner of the tournament.
        # The winner is the one with the minimum MSE score.
        # The `key=lambda item: item[1]` tells the `min` function to look at the second element of each tuple (the mse_score) for the comparison.
        winner = min(tournament_entrants, key=lambda item: item[1])
        
        # 3. Return the winner's tree.
        # The `winner` variable is a tuple like (['+', 'S1', 'c'], 0.123), so we return the first element, which is the equation tree itself.
        return winner[0]
    

    def fit(self,
            conditions: Union[pd.DataFrame, np.ndarray],
            observations: Union[pd.DataFrame, np.ndarray]):
        pass

    def predict(self,
                conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

if __name__ == "__main__":
    # Example usage of the NutsTheorists class
    theorist = NutsTheorists()
    random_tree = theorist._create_random_tree(max_depth=3)
    random_tree2 = theorist._create_random_tree(max_depth=3)

    next_gen = generate_next_generation([random_tree, random_tree2], pop_size=4)

    for i, tree in enumerate(next_gen):
        print(f"Next generation tree {i+1}:         ", tree)



class SimpleLinearTheorist(BaseEstimator):
    """
    A simple theorist that fits a linear equation: y = a*x + b
    """

    def __init__(self):
        self.model = LinearRegression()
        self.coef_ = None
        self.intercept_ = None

    def fit(self, conditions, observations):
        self.model.fit(conditions, observations)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

    def predict(self, conditions):
        return self.model.predict(conditions)

    def print_eqn(self):
        # Handles single or multi-output
        if hasattr(self.coef_, "shape") and len(self.coef_.shape) > 1:
            eqns = []
            for i, (coef, intercept) in enumerate(zip(self.coef_, self.intercept_)):
                terms = " + ".join([f"{c:.3f}*x{j+1}" for j, c in enumerate(coef)])
                eqns.append(f"y{i+1} = {terms} + {intercept:.3f}")
            return "\n".join(eqns)
        else:
            terms = " + ".join([f"{c:.3f}*x{j+1}" for j, c in enumerate(self.coef_)])
            return f"y = {terms} + {self.intercept_:.3f}"

class UniversalTheorist(BaseEstimator):
    """
    Automatically fits a power law for single feature,
    or a log-ratio law for two features.
    """

    def __init__(self):
        self.model_type = None
        self.params_ = None

    def _power_law(self, x, a, b, c):
        return a * np.power(x, b) + c

    def _log_ratio(self, X, a, b):
        S1 = X[:, 0]
        S2 = X[:, 1]
        return a * np.log(S1 / S2) + b

    def fit(self, conditions, observations):
        if isinstance(conditions, pd.DataFrame):
            X = conditions.values
        else:
            X = np.array(conditions)
        y = np.array(observations).flatten()
        n_features = X.shape[1] if X.ndim > 1 else 1

        if n_features == 1:
            self.model_type = "power"
            x = X.flatten()
            popt, _ = curve_fit(self._power_law, x, y, p0=[1, 1, 0], maxfev=10000)
            self.params_ = popt
        elif n_features == 2:
            self.model_type = "logratio"
            popt, _ = curve_fit(self._log_ratio, X, y, p0=[1, 0], maxfev=10000)
            self.params_ = popt
        else:
            raise ValueError("UniversalTheorist only supports 1 or 2 input features.")

    def predict(self, conditions):
        if isinstance(conditions, pd.DataFrame):
            X = conditions.values
            index = conditions.index
        else:
            X = np.array(conditions)
            index = None
        n_features = X.shape[1] if X.ndim > 1 else 1

        if self.model_type == "power":
            x = X.flatten()
            a, b, c = self.params_
            y_pred = self._power_law(x, a, b, c)
        elif self.model_type == "logratio":
            a, b = self.params_
            y_pred = self._log_ratio(X, a, b)
        else:
            raise ValueError("Model not fitted or unsupported feature count.")

        if index is not None:
            return pd.Series(y_pred, index=index)
        else:
            return pd.Series(y_pred)

    def print_eqn(self):
        if self.model_type == "power":
            a, b, c = self.params_
            eqn = f"y = {a:.3f} * x^{b:.3f} + {c:.3f}"
            print(f"Equation: {eqn}")
            return eqn
        elif self.model_type == "logratio":
            a, b = self.params_
            eqn = f"y = {a:.3f} * ln(x1/x2) + {b:.3f}"
            print(f"Equation: {eqn}")
            return eqn
        else:
            return "No model fitted."

