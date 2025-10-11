import numpy as np
import pandas as pd
from autora.theorist.nuts import NutsRegressor

def test_theorist_recovery_test():
    """
    Tests that the NutsRegressor can recover a simple, known linear relationship.
    This is a more rigorous test of the theorist's discovery capability.
    """
    # 1. Define a known ground truth: y = 2.5 * x1 + 1.5
    true_slope = 2.5
    true_intercept = 1.5
    X_train = pd.DataFrame({"x1": np.linspace(0, 10, 20)})
    y_train = pd.DataFrame({"y": true_slope * X_train["x1"] + true_intercept})

    X_test = pd.DataFrame({"x1": np.linspace(11, 15, 10)})
    y_test_true = true_slope * X_test["x1"] + true_intercept

    # 2. Instantiate the theorist with parameters suitable for finding this equation
    theorist = NutsRegressor(
        population_size=100,
        n_generation=15,
        n_constants=2,  # Allow for two constants (slope and intercept)
        complexity_penalty=0.01,
        verbose=False
    )

    # 3. Fit the model to the training data
    theorist.fit(X_train, y_train)

    # 4. Make predictions on the test data
    predictions = theorist.predict(X_test)

    # 5. Assert that the predictions are very close to the true values
    # We use np.allclose to check for approximate equality, allowing for small errors.
    assert np.allclose(predictions, y_test_true.values, atol=0.1), \
        "Theorist failed to recover the linear relationship accurately."
