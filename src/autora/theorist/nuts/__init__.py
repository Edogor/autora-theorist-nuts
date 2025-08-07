"""
Example Theorist
"""
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


 
class ExampleRegressor(BaseEstimator):
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

    def __init__(self):
        pass

    def fit(self,
            conditions: Union[pd.DataFrame, np.ndarray],
            observations: Union[pd.DataFrame, np.ndarray]):
        pass

    def predict(self,
                conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

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

