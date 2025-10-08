"""
Simple and quick demonstration of the Island Model feature.
This runs fast to show the key concepts without taking too long.
"""

import numpy as np
import pandas as pd

from autora.theorist.nuts import Island, NutsTheorists

# Set random seed for reproducibility
np.random.seed(42)

print("\n" + "=" * 80)
print("NUTS THEORIST - ISLAND MODEL QUICK DEMO")
print("=" * 80)

# Generate simple linear data
print("\nGenerating test data: y = 3*x + 2")
X = pd.DataFrame({"x": np.linspace(0, 10, 30)})
y = pd.DataFrame({"y": 3 * X["x"] + 2})

print("\nCreating 3 islands with different strategies:")
islands = [
    Island(
        name="Conservative",
        population_size=40,
        mutation_rate=0.1,
        tournament_size=10,
        elitism=10,
        operator_weights={"unary": 1.5, "binary": 1.0},
    ),
    Island(
        name="Balanced",
        population_size=40,
        mutation_rate=0.2,
        tournament_size=15,
        elitism=8,
        operator_weights={"unary": 1.0, "binary": 1.0},
    ),
    Island(
        name="Aggressive",
        population_size=40,
        mutation_rate=0.4,
        tournament_size=20,
        elitism=3,
        operator_weights={"unary": 0.8, "binary": 1.5},
    ),
]

print("\nIsland configurations:")
for island in islands:
    print(f"  - {island.name}: mutation_rate={island.mutation_rate}, elitism={island.elitism}")

# Run evolution with island model
print("\nRunning evolution with island model (5 generations)...")
theorist = NutsTheorists(
    n_generation=5,
    use_island_model=True,
    islands=islands,
    n_constants=2,
    verbose=False,  # Quiet mode for cleaner output
)

theorist.fit(X, y)

# Show results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nBest Equation Found: {theorist.best_equation}")
print(f"Best MSE: {theorist.best_fitness:.6f}")
print(f"Winning Island: {theorist.island_model.global_best_island}")

# Show island rankings
print("\nIsland Rankings:")
rankings = theorist.get_island_summary()["rankings"]
for rank, stats in enumerate(rankings, 1):
    status = "✓" if stats["alive"] else "✗"
    print(f"  {rank}. {stats['name']:<15} MSE: {stats['best_fitness']:.6f}  {status}")

print("\n" + "=" * 80)
print("KEY BENEFITS OF ISLAND MODEL")
print("=" * 80)
print(
    """
✓ Multiple strategies evolve in parallel
✓ Automatic selection of best approach
✓ Better exploration of solution space
✓ Easy to customize island behaviors
✓ Tracks which parameters work best

Future extensions are easy to add:
  - Island migration (share top solutions between islands)
  - Dynamic lifecycle (remove weak islands, create new ones)
  - Hybrid islands (combine strategies from successful islands)
"""
)

print("=" * 80)
print("Demo complete! See docs/island_model.md for full documentation.")
print("=" * 80)
