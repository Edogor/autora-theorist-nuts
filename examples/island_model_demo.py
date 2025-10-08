"""
Demo script showcasing the Island Model feature for NUTS Theorist.

This demonstrates:
1. Creating custom islands with different parameters
2. Running evolution with multiple islands
3. Viewing island performance and rankings
4. Comparing island model vs standard approach
"""

import numpy as np
import pandas as pd

from autora.theorist.nuts import Island, NutsTheorists

# Set random seed for reproducibility
np.random.seed(42)


def demo_basic_island_model():
    """Basic demonstration of island model with simple synthetic data."""
    print("=" * 80)
    print("DEMO 1: Basic Island Model")
    print("=" * 80)

    # Generate simple polynomial data: y = 2*x^2 + 3*x + 1
    X = pd.DataFrame({"x": np.linspace(-5, 5, 50)})
    y = pd.DataFrame({"y": 2 * X["x"] ** 2 + 3 * X["x"] + 1})

    # Create islands with different "cultures" (parameter settings)
    islands = [
        Island(
            name="Conservative",
            population_size=80,
            mutation_rate=0.1,
            tournament_size=15,
            elitism=20,
            operator_weights={"unary": 1.5, "binary": 1.0, "crossover": 1.2, "mutation": 0.8},
        ),
        Island(
            name="Balanced",
            population_size=80,
            mutation_rate=0.2,
            tournament_size=20,
            elitism=10,
            operator_weights={"unary": 1.0, "binary": 1.0, "crossover": 1.0, "mutation": 1.0},
        ),
        Island(
            name="Aggressive",
            population_size=80,
            mutation_rate=0.4,
            tournament_size=25,
            elitism=5,
            operator_weights={"unary": 0.8, "binary": 1.5, "crossover": 0.8, "mutation": 1.5},
        ),
    ]

    # Create theorist with island model
    theorist = NutsTheorists(
        n_generation=10,
        verbose=True,
        use_island_model=True,
        islands=islands,
        n_constants=3,
        early_stopping_rounds=None,  # disable early stopping for demo
    )

    print("\nFitting with Island Model...")
    theorist.fit(X, y)

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Best Equation: {theorist.best_equation}")
    print(f"Best MSE: {theorist.best_fitness:.6f}")
    print(f"Winning Island: {theorist.island_model.global_best_island}")


def demo_custom_heuristics():
    """Demonstrate islands with different operator preferences."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Islands with Different Operator Heuristics")
    print("=" * 80)

    # Generate exponential data: y = exp(x/2)
    X = pd.DataFrame({"x": np.linspace(0, 3, 30)})
    y = pd.DataFrame({"y": np.exp(X["x"] / 2)})

    islands = [
        Island(
            name="UnaryLovers",
            population_size=60,
            mutation_rate=0.2,
            tournament_size=20,
            elitism=10,
            operator_weights={"unary": 3.0, "binary": 0.5},  # Prefers unary ops like log, exp
        ),
        Island(
            name="BinaryFans",
            population_size=60,
            mutation_rate=0.2,
            tournament_size=20,
            elitism=10,
            operator_weights={"unary": 0.5, "binary": 3.0},  # Prefers binary ops like +, *, /
        ),
        Island(
            name="MutationHeavy",
            population_size=60,
            mutation_rate=0.5,  # High mutation
            tournament_size=20,
            elitism=5,
            operator_weights={"unary": 1.0, "binary": 1.0, "mutation": 2.0},
        ),
    ]

    theorist = NutsTheorists(
        n_generation=8,
        verbose=True,
        use_island_model=True,
        islands=islands,
        n_constants=2,
    )

    print("\nFitting with custom heuristics...")
    theorist.fit(X, y)

    print("\nIsland Rankings:")
    rankings = theorist.island_model.get_island_rankings()
    for rank, stats in enumerate(rankings, 1):
        print(f"{rank}. {stats['name']}: MSE = {stats['best_fitness']:.6f}")


def demo_island_statistics():
    """Show detailed island statistics and performance tracking."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Island Performance Statistics")
    print("=" * 80)

    # Generate linear data
    X = pd.DataFrame({"x": np.linspace(0, 10, 40)})
    y = pd.DataFrame({"y": 5 * X["x"] - 2})

    # Create diverse islands
    islands = [
        Island("Fast", population_size=50, mutation_rate=0.15, elitism=15),
        Island("Medium", population_size=70, mutation_rate=0.25, elitism=10),
        Island("Slow", population_size=90, mutation_rate=0.35, elitism=5),
    ]

    theorist = NutsTheorists(
        n_generation=6,
        verbose=False,  # Less verbose for cleaner output
        use_island_model=True,
        islands=islands,
        n_constants=2,
    )

    theorist.fit(X, y)

    # Print detailed summary
    theorist.print_island_summary()

    # Show evolution history for each island
    print("\nIsland Evolution History:")
    for island in theorist.island_model.islands:
        print(f"\n{island.name}:")
        print(f"  Final Fitness: {island.best_fitness:.6f}")
        print(f"  Generations: {len(island.generation_history)}")
        if island.generation_history:
            fitness_vals = [f"{h['best_fitness']:.4f}" for h in island.generation_history[:5]]
            print(f"  Fitness Progression: {fitness_vals}")


def demo_vs_standard():
    """Compare island model performance vs standard approach."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Island Model vs Standard Approach")
    print("=" * 80)

    # Generate power law data
    X = pd.DataFrame({"x": np.linspace(1, 5, 35)})
    y = pd.DataFrame({"y": X["x"] ** 1.5})

    print("\n--- Standard Approach ---")
    standard = NutsTheorists(
        population_size=200,  # Same total population as island model
        n_generation=6,
        verbose=False,
        use_island_model=False,
        n_constants=2,
    )
    standard.fit(X, y)
    print(f"Best MSE: {standard.best_fitness:.6f}")
    print(f"Best Equation: {standard.best_equation}")

    print("\n--- Island Model Approach ---")
    islands = [
        Island("Island1", population_size=50, mutation_rate=0.15),
        Island("Island2", population_size=50, mutation_rate=0.25),
        Island("Island3", population_size=50, mutation_rate=0.35),
        Island("Island4", population_size=50, mutation_rate=0.20),
    ]

    island_theorist = NutsTheorists(
        n_generation=6,
        verbose=False,
        use_island_model=True,
        islands=islands,
        n_constants=2,
    )
    island_theorist.fit(X, y)
    print(f"Best MSE: {island_theorist.best_fitness:.6f}")
    print(f"Best Equation: {island_theorist.best_equation}")
    print(f"Best Island: {island_theorist.island_model.global_best_island}")


def demo_default_islands():
    """Show automatic creation of default islands."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Using Default Islands")
    print("=" * 80)
    print("When use_island_model=True but no islands are specified,")
    print("the system creates 4 default islands with different parameters.")

    X = pd.DataFrame({"x": np.linspace(-3, 3, 30)})
    y = pd.DataFrame({"y": X["x"] ** 2 + 1})

    theorist = NutsTheorists(
        n_generation=5,
        verbose=False,
        use_island_model=True,  # No islands specified
        n_constants=2,
    )

    theorist.fit(X, y)

    print(f"\nNumber of default islands created: {len(theorist.island_model.islands)}")
    print("\nDefault islands:")
    for island in theorist.island_model.islands:
        print(
            f"  - {island.name}: pop={island.population_size}, "
            f"mut_rate={island.mutation_rate}, elitism={island.elitism}"
        )

    theorist.print_island_summary()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "NUTS THEORIST - ISLAND MODEL DEMO" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")

    # Run all demos
    demo_basic_island_model()
    demo_custom_heuristics()
    demo_island_statistics()
    demo_vs_standard()
    demo_default_islands()

    print("\n\n" + "=" * 80)
    print("DEMOS COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Creating custom islands with different parameters")
    print("  ✓ Operator preference heuristics per island")
    print("  ✓ Performance tracking and island rankings")
    print("  ✓ Automatic best island identification")
    print("  ✓ Default islands for easy setup")
    print("\nFuture Enhancements (easy to add):")
    print("  - Island migration (sharing top individuals)")
    print("  - Dynamic island lifecycle (killing weak islands)")
    print("  - Creating new hybrid islands from successful ones")
    print("=" * 80)
