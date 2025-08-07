"""
Example Theorist
"""
from typing import Union
import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator



class NutsTheorists(BaseEstimator):
    """
    Include inline mathematics in docstring \\(x < 1\\) or $c = 3$
    or block mathematics:

    \\[
        x + 1 = 3
    \\]


    $$
    y + 1 = 4
    $$

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
    def fit(self,
            conditions: Union[pd.DataFrame, np.ndarray],
            observations: Union[pd.DataFrame, np.ndarray]):
        pass

    def predict(self,
                conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass
