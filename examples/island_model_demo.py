"""
Demonstration of the Island Model Feature in NutsTheorists
===========================================================

This script demonstrates how to use the island model feature and compares
its performance with the standard single-population model.

The island model is a parallel genetic algorithm technique where multiple
populations (islands) evolve independently with periodic migration of best
individuals between islands.
"""

import numpy as np
import pandas as pd
from autora.theorist.nuts import NutsTheorists

# Create test data: y = 2*x + 3 (simple linear relationship)
np.random.seed(42)
X = pd.DataFrame({'x': np.linspace(0, 10, 50)})
y = pd.DataFrame({'y': 2 * X['x'] + 3 + np.random.normal(0, 0.5, 50)})

print("=" * 80)
print("ISLAND MODEL DEMONSTRATION")
print("=" * 80)
print(f"\nTest Data: y = 2*x + 3 (with Gaussian noise)")
print(f"Number of samples: {len(X)}")

# Test 1: Standard single population model
print("\n" + "-" * 80)
print("Model 1: Standard Model (Single Population)")
print("-" * 80)
print("Configuration:")
print("  - Population size: 100")
print("  - Generations: 10")
print("  - Islands: 1 (standard model)")

standard_model = NutsTheorists(
    n_islands=1,
    population_size=100,
    n_generation=10,
    verbose=False,
    early_stopping_rounds=5
)
standard_model.fit(X, y)

print(f"\nResults:")
print(f"  Best equation: {standard_model.best_equation}")
print(f"  Best MSE: {standard_model.best_fitness:.4f}")

# Test 2: Island model with migration
print("\n" + "-" * 80)
print("Model 2: Island Model (Multiple Populations with Migration)")
print("-" * 80)
print("Configuration:")
print("  - Population size: 100")
print("  - Generations: 10")
print("  - Islands: 3")
print("  - Migration rate: 0.15 (15% of each island's population)")
print("  - Migration interval: 3 (migration every 3 generations)")

island_model = NutsTheorists(
    n_islands=3,
    migration_rate=0.15,
    migration_interval=3,
    population_size=100,
    n_generation=10,
    verbose=False,
    early_stopping_rounds=5
)
island_model.fit(X, y)

print(f"\nResults:")
print(f"  Best equation: {island_model.best_equation}")
print(f"  Best MSE: {island_model.best_fitness:.4f}")

# Compare results
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"Standard Model MSE:  {standard_model.best_fitness:.4f}")
print(f"Island Model MSE:    {island_model.best_fitness:.4f}")

if island_model.best_fitness < standard_model.best_fitness:
    improvement = ((standard_model.best_fitness - island_model.best_fitness) / 
                   standard_model.best_fitness * 100)
    print(f"\nIsland model achieved {improvement:.2f}% improvement over standard model!")
elif standard_model.best_fitness < island_model.best_fitness:
    print(f"\nStandard model performed better in this run.")
else:
    print(f"\nBoth models achieved similar performance.")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
The island model offers several advantages:

1. **Diversity Maintenance**: Multiple independent populations maintain
   genetic diversity better than a single large population.

2. **Parallel Exploration**: Different islands can explore different regions
   of the solution space simultaneously.

3. **Migration Benefits**: Periodic migration spreads good genetic material
   while preserving local exploration.

4. **Robustness**: Helps avoid premature convergence to local optima.

The actual performance gain depends on:
- The complexity of the problem
- Population size and number of islands
- Migration rate and interval settings
- Random initialization

Run this script multiple times to see the variability in results!
""")
