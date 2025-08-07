# NOTE:
# max_depth
# n_generation
# TODO:
# different values for constant (random ?)

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

    def __init__(self, population_size=100, n_generation=50, mutation_rate=0.1, tournament_size=3):
        self.population_size = population_size
        self.n_generation = n_generation
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = 2  # Number of best trees to carry over unchanged

        #Attributes to store the final result
        self.best_equation = None
        self.best_params = None
        self.best_fitness = -1
        self.result_list = []
        self.UNARY_OPS = ['np.log', 'np.exp']
        self.BINARY_OPS = ['+', '-', '*', '/', 'np.power']

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
        
    def _translate_tree_to_callable(self, eq_str, var_names, constant_value=1.0):
        eq_str = eq_str.replace('^', '**')
        eq_str = eq_str.replace('exp', 'np.exp').replace('log', 'np.log')


        lambda_str = f"lambda {', '.join(var_names)}: {eq_str}"
        return eval(lambda_str, {"np": np})

    def _evaluate_tree_mse(self, tree, conditions, observations, constant_value=1.0):
        """
        Translates a tree into a function, evaluates it on the conditions,
        and computes the MSE against observations.

        Returns:
            tuple: (tree, mse)
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


    def generate_next_generation(self, top_k_trees, pop_size=1000, mutation_rate=0.2, max_depth=3, elitism=10):
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
        def get_random_subtree(tree):
            """
            Returns a randomly selected subtree, its parent, and index in parent.
            """
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

        def crossover(tree1, tree2):
            print("Crossover between trees:")
            print(tree1)
            print(tree2)
#            def get_random_subtree(tree):
#                if not isinstance(tree, list):
#                    return tree, None, None
#                idx = random.randint(1, len(tree)-1)
#                return tree[idx], tree, idx

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
                        return random.choice(self.TERMINALS)
                    return node

                # Subtree mutation
                if random.random() < mutation_rate:
                    return self._create_random_tree(max_depth)

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
     
        
    def tree_to_function(self, tree):
        """
        Converts a nested list expression like ['+', 'S1', ['*', 'S2', 'c']]
        into a Python function that can be evaluated with input data.
        """

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

    
#    def fitness_function(self, expression_func, conditions, observations):
#        try:
#            preds = expression_func(conditions)
#            return mean_squared_error(observations, preds)
#        except Exception:
#            return -float("inf") 

    def append_tree_score(self, tree, mse):
        self.result_list.append((tree, mse))
        return self.result_list


    def _tournament(self, population_with_scores, k=50):
        """
        Selects the top-k individuals from the population using tournament selection.

        Args:
            population_with_scores (list): List of tuples (tree, mse_score)
            k (int): Number of top individuals to return

        Returns:
            list: List of top-k (tree, mse_score) tuples
        """
        # 1. Randomly select individuals for the tournament
        tournament_entrants = random.sample(population_with_scores, min(len(population_with_scores), self.tournament_size * k))

        # 2. Sort by MSE (ascending = better)
        sorted_entrants = sorted(tournament_entrants, key=lambda item: item[1])

        # 3. Return top-k (tree, mse_score) pairs
        return sorted_entrants[:k]


    def fit(self, conditions: pd.DataFrame, observations: pd.DataFrame):
        """
        Runs the genetic algorithm to find the best equation.
        """
        # 1. INITIALIZATION (Happens once)
        # a. Adapt terminals to the specific problem's data

        # 1. Normalize input
        if isinstance(conditions, pd.DataFrame):
            conditions = conditions.copy()
        elif isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=[f'x{i+1}' for i in range(conditions.shape[1])])

        # 2. Set variable names and terminals before using them anywhere
        self.var_names = conditions.columns.tolist()
        self.TERMINALS = [
            str(round(random.uniform(0.1, 5.0), 3)) for _ in range(5)
        ] + self.var_names

        # 3. Create initial population (now TERMINALS is available!)
        y_true = observations.values.ravel()
        population = [self._create_random_tree(5) for _ in range(self.population_size)]
            

        # b. Create the initial random population (Generation 0)
        population = [self._create_random_tree(5) for _ in range(self.population_size)]
        self.initial_population = population.copy()
        # --- 2. THE MAIN EVOLUTION LOOP ---
        for generation in range(self.n_generation):
            # a. EVALUATION: Score every individual in the current population.
            pop_with_scores = []
            for tree in population:
                #self, tree, conditions, observations, constant_value=1.0)
                tree, fitness = self._evaluate_tree_mse(tree, conditions, observations, constant_value=1.0)
                pop_with_scores.append((tree, fitness))

            # b. TRACK THE BEST: Sort by fitness and check for a new best-ever solution.
            # best_trees = self._tournament(pop_with_scores)
            top_k = self._tournament(pop_with_scores, k=50)
            top_k_trees = [tree for tree, _ in top_k]
            population = self.generate_next_generation(top_k_trees, pop_size=self.population_size)

            #pop_with_scores.sort(key=lambda item: item[1], reverse=True)
            
            #if pop_with_scores[0][1] > self.best_fitness:
             #   self.best_fitness = pop_with_scores[0][1]
              #  self.best_equation_ = pop_with_scores[0][0]
               # print(f"Gen {generation+1}: New best fitness = {self.best_fitness:.4f}")

            # c. REPRODUCTION: Create the next generation's population.



#            # Crossover & Mutation: Fill the rest of the population.
#            while len(new_population) < self.population_size:
#                parent1 = self._tournament(pop_with_scores)
#                parent2 = self._tournament(pop_with_scores)
#                
#                child = self.crossover(parent1, parent2)
#                child = self._mutate(child)
#                
#                new_population.append(child)

#            # d. REPLACEMENT: The new generation becomes the current population.
#            population = new_population
        best_equation = top_k[0][0]
        best_fitness = top_k[0][1]
        self.best_equation = self._tree_translate(best_equation)
        self.best_fitness = best_fitness

        print("\nEvolution finished.")
        print(f"Best equation found: {self.best_equation}")
        print(f"Best fitness (-MSE): {self.best_fitness}")
        
        return self


    def predict(self, conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.best_equation is None:
            raise ValueError("No equation available. Did you forget to call fit()?")

        if isinstance(conditions, np.ndarray):
            conditions = pd.DataFrame(conditions, columns=self.var_names)

        eq_str = self._tree_translate(self.best_equation)
        func = self._translate_tree_to_callable(eq_str, self.var_names, constant_value=1.0)

        preds = np.array([func(*row) for row in conditions.values]).reshape(-1, 1)
        return preds
    
    def print_eqn(self):
        if self.best_equation is None:
            print("No equation available. Did you forget to call fit()?")
            return
        
        eq_str = self._tree_translate(self.best_equation)
        print(f"Best equation: {eq_str}")
        return eq_str

if __name__ == "__main__":
    theorist = NutsTheorists()

    # Manually set var names and terminals for testing
    theorist.var_names = ['S1', 'S2']  # or just ['S'] depending on your case
    theorist.TERMINALS = [
        str(round(random.uniform(0.1, 5.0), 3)) for _ in range(5)
    ] + theorist.var_names

    # Generate and test trees
    random_tree = theorist._create_random_tree(max_depth=3)
    random_tree2 = theorist._create_random_tree(max_depth=3)

    next_gen = theorist.generate_next_generation([random_tree, random_tree2], pop_size=4)

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

