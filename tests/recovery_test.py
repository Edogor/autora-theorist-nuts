import pytest
import numpy as np
import pandas as pd
from src.autora.theorist.nuts import NutsTheorists

@pytest.fixture
def seed():
    """ Ensures that the results are the same each time the tests are run. """
    import random
    random.seed(180)
    np.random.seed(180)
    return

def test_theorist_recovery_test(seed):
    """
    Tests that the NutsTheorists can recover a simple, known linear relationship.
    """
    # 1. Define a known ground truth: y = 2.5 * x1 + 1.5
    true_slope = 2.5
    true_intercept = 1.5
    X_train = pd.DataFrame({"x1": np.linspace(0, 10, 20)})
    y_train = pd.DataFrame({"y": true_slope * X_train["x1"] + true_intercept})

    X_test = pd.DataFrame({"x1": np.linspace(11, 15, 10)})
    y_test_true = true_slope * X_test["x1"] + true_intercept

    # 2. Instantiate the theorist with slightly larger parameters for a more robust test
    theorist = NutsTheorists(
        population_size=100,      # Increased from 50
        n_generation=20,          # Increased from 10
        n_constants=2,
        complexity_penalty=0.01,
        verbose=False,
        constant_grid=np.linspace(-3, 3, 15) # Slightly larger grid
    )

    # 3. Fit the model to the training data
    theorist.fit(X_train, y_train)

    # 4. Make predictions on the test data
    predictions = theorist.predict(X_test)

    # 5. Assert that the predictions are very close to the true values
    # We keep a reasonable tolerance (atol=0.2)
    assert np.allclose(predictions, y_test_true.values, atol=0.2), \
        "Theorist failed to recover the linear relationship accurately."

