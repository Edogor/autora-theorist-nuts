import numpy as np
import pandas as pd
# Use the correct class name 'NutsTheorists'
from autora.theorist.nuts import NutsTheorists

def test_theorist_smoke_test():
    """
    Tests that the NutsTheorists can be instantiated, fit, and predict without errors.
    """
    # 1. Create a simple synthetic dataset
    X_train = pd.DataFrame({"x1": np.linspace(-5, 5, 10)})
    y_train = pd.DataFrame({"y": 3.0 * X_train["x1"] + 2.0})

    X_test = pd.DataFrame({"x1": np.linspace(6, 8, 5)})

    # 2. Instantiate the theorist with minimal parameters for a quick run
    theorist = NutsTheorists(
        population_size=10,
        n_generation=3,
        verbose=False  # Keep the test output clean
    )

    # 3. Fit the model
    theorist.fit(X_train, y_train)

    # 4. Make predictions
    predictions = theorist.predict(X_test)

    # 5. Assert that the predictions have the correct shape
    assert predictions.shape == (5, 1), "Prediction output shape is incorrect."
    
    # 6. Check that a valid equation string is produced
    equation = theorist.print_eqn()
    assert isinstance(equation, str), "print_eqn should return a string."

