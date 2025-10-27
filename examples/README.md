# Island Model Examples

This directory contains examples demonstrating the island model feature in NutsTheorists.

## island_model_demo.py

A simple demonstration script that compares the standard single-population genetic programming approach with the island model approach.

### Running the Demo

```bash
python examples/island_model_demo.py
```

### What is the Island Model?

The island model is a parallel genetic algorithm technique where:
- Multiple populations (islands) evolve independently
- Best individuals periodically migrate between islands
- This maintains genetic diversity and helps avoid premature convergence

### Key Parameters

- **n_islands**: Number of separate populations (default: 1 for standard model)
- **migration_rate**: Fraction of population to migrate (e.g., 0.1 = 10%)
- **migration_interval**: Generations between migrations (e.g., 5 = migrate every 5 generations)

### Example Usage

```python
from autora.theorist.nuts import NutsTheorists

# Standard single-population model
standard_model = NutsTheorists(n_islands=1)

# Island model with 3 islands
island_model = NutsTheorists(
    n_islands=3,
    migration_rate=0.1,
    migration_interval=5
)

# Both models use the same fit/predict interface
island_model.fit(X_train, y_train)
predictions = island_model.predict(X_test)
```

## See Also

- Check `challenge/theorist_benchmarking.ipynb` for benchmark comparisons
- The notebook includes side-by-side comparisons on real cognitive models
