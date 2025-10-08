# Island Model for NUTS Theorist

The Island Model is a genetic programming technique that maintains multiple independent populations (islands) that evolve separately with different parameters and operator preferences. This approach can lead to better exploration of the solution space and often finds better solutions than a single large population.

## Overview

The Island Model implementation includes:

- **Island Class**: Represents a single island with its own population and parameters
- **IslandModel Class**: Manages multiple islands and tracks global performance
- **NutsTheorists Integration**: Seamlessly works with existing NutsTheorists API

## Quick Start

### Basic Usage with Default Islands

The easiest way to use the island model is to simply enable it:

```python
from autora.theorist.nuts import NutsTheorists
import pandas as pd
import numpy as np

# Create your data
X = pd.DataFrame({'x': np.linspace(0, 10, 50)})
y = pd.DataFrame({'y': 2 * X['x'] + 1})

# Enable island model with default islands
theorist = NutsTheorists(
    n_generation=20,
    use_island_model=True,  # This creates 4 default islands
    verbose=True
)

theorist.fit(X, y)

# View island performance
theorist.print_island_summary()
```

### Custom Islands with Different Parameters

Create islands with custom parameters to explore different evolutionary strategies:

```python
from autora.theorist.nuts import NutsTheorists, Island

# Create custom islands with different "cultures"
islands = [
    Island(
        name="Conservative",
        population_size=100,
        mutation_rate=0.1,      # Low mutation
        tournament_size=15,
        elitism=20,             # High elitism
    ),
    Island(
        name="Balanced",
        population_size=100,
        mutation_rate=0.2,      # Medium mutation
        tournament_size=20,
        elitism=10,             # Medium elitism
    ),
    Island(
        name="Aggressive",
        population_size=100,
        mutation_rate=0.4,      # High mutation
        tournament_size=25,
        elitism=5,              # Low elitism
    ),
]

theorist = NutsTheorists(
    n_generation=20,
    use_island_model=True,
    islands=islands,
    verbose=True
)

theorist.fit(X, y)
```

### Operator Preferences (Heuristics)

Each island can have different operator preferences, allowing some islands to favor certain types of operations:

```python
from autora.theorist.nuts import Island

islands = [
    Island(
        name="UnaryLovers",
        population_size=80,
        mutation_rate=0.2,
        operator_weights={
            'unary': 3.0,      # Strongly prefers unary operators (log, exp)
            'binary': 0.5,     # Less likely to use binary operators
            'crossover': 1.0,
            'mutation': 1.0
        }
    ),
    Island(
        name="BinaryFans",
        population_size=80,
        mutation_rate=0.2,
        operator_weights={
            'unary': 0.5,      # Less likely to use unary operators
            'binary': 3.0,     # Strongly prefers binary operators (+, -, *, /)
            'crossover': 1.0,
            'mutation': 1.0
        }
    ),
    Island(
        name="MutationHeavy",
        population_size=80,
        mutation_rate=0.5,     # High mutation rate
        operator_weights={
            'unary': 1.0,
            'binary': 1.0,
            'crossover': 0.8,
            'mutation': 2.0    # Prefers mutation over crossover
        }
    ),
]

theorist = NutsTheorists(
    n_generation=20,
    use_island_model=True,
    islands=islands
)

theorist.fit(X, y)
```

## Island Performance Tracking

The island model automatically tracks which islands perform best:

```python
# After fitting
theorist.fit(X, y)

# Get island rankings
summary = theorist.get_island_summary()
print(f"Best island: {summary['global_best_island']}")
print(f"Best fitness: {summary['global_best_fitness']}")

# Print detailed summary
theorist.print_island_summary()

# Output example:
# ================================================================================
# ISLAND MODEL SUMMARY
# ================================================================================
# 
# Global Best Island: Conservative
# Global Best Fitness: 0.001234
# 
# Island Rankings:
# Rank  Island              Fitness        Pop Size    Mut Rate    Status    
# --------------------------------------------------------------------------------
# 1     Conservative        0.001234       100         0.100       ALIVE     
# 2     Balanced            0.002456       100         0.200       ALIVE     
# 3     Aggressive          0.005678       100         0.400       ALIVE     
# ================================================================================
```

## API Reference

### Island Class

```python
Island(
    name: str,                              # Unique identifier for the island
    population_size: int = 100,             # Number of individuals
    mutation_rate: float = 0.2,             # Probability of mutation
    tournament_size: int = 20,              # Tournament selection size
    elitism: int = 10,                      # Number of elite individuals preserved
    operator_weights: Dict[str, float] = None  # Operator preferences
)
```

**Operator Weights Keys:**
- `'unary'`: Weight for unary operators (np.log, np.exp)
- `'binary'`: Weight for binary operators (+, -, *, /, np.power)
- `'crossover'`: Weight for crossover operations
- `'mutation'`: Weight for mutation operations

### IslandModel Class

```python
IslandModel(islands: List[Island])
```

**Methods:**
- `get_best_island()`: Returns the island with best fitness
- `update_global_best()`: Updates global best from all islands
- `get_island_rankings()`: Returns list of island statistics sorted by fitness
- `print_island_summary()`: Prints detailed island performance summary
- `kill_weak_islands(fitness_threshold)`: Marks islands with poor fitness as dead
- `get_migration_candidates(n_islands, n_trees)`: Gets top trees from best islands for migration

### NutsTheorists with Island Model

```python
NutsTheorists(
    # Standard parameters...
    use_island_model: bool = False,    # Enable island model
    islands: List[Island] = None,      # Custom islands (optional)
    # ... other parameters
)
```

**Additional Methods:**
- `get_island_summary()`: Returns dictionary with island statistics
- `print_island_summary()`: Prints island performance summary

## Design for Future Extensions

The island model is designed to easily support future enhancements:

### 1. Dynamic Island Lifecycle

Islands with poor performance can be automatically removed:

```python
# Already implemented in IslandModel class
island_model.kill_weak_islands(fitness_threshold=10.0)
```

### 2. Island Migration

Share top individuals between islands (framework ready, easy to add):

```python
# Get migration candidates (already implemented)
candidates = island_model.get_migration_candidates(n_islands=2, n_trees=5)

# Future: Inject these candidates into other islands
# for target_island in other_islands:
#     target_island.population.extend(candidates)
```

### 3. Hybrid Islands

Create new islands from successful ones (framework ready):

```python
# Future implementation:
# best_island = island_model.get_best_island()
# new_island = Island(
#     name="Hybrid",
#     population_size=best_island.population_size,
#     mutation_rate=best_island.mutation_rate * 0.8,  # Slightly different
#     ...
# )
# new_island.population = copy.deepcopy(best_island.population[:20])
# island_model.islands.append(new_island)
```

## Advantages of Island Model

1. **Diverse Exploration**: Different islands explore different regions of solution space
2. **Parameter Robustness**: No need to tune a single set of parameters perfectly
3. **Parallel Evolution**: Islands evolve independently (can be parallelized in future)
4. **Adaptive Strategy**: Different strategies compete, best one wins
5. **Prevents Premature Convergence**: Multiple populations maintain diversity

## When to Use Island Model

**Use Island Model When:**
- You want to explore multiple evolutionary strategies simultaneously
- You're unsure about optimal parameter settings
- You want better solution quality and are willing to wait longer
- You need to handle complex, multimodal fitness landscapes

**Use Standard Approach When:**
- You need fast results with limited computational budget
- You have well-tuned parameters for your problem
- Your problem is relatively simple
- You need backwards compatibility with existing code

## Backwards Compatibility

The island model is completely backwards compatible. Existing code continues to work without modification:

```python
# This still works exactly as before
theorist = NutsTheorists(population_size=500, n_generation=30)
theorist.fit(X, y)
```

## Examples

See `examples/island_model_demo.py` for comprehensive demonstrations of:
- Basic island model usage
- Custom heuristics and operator preferences
- Performance tracking and statistics
- Comparison with standard approach
- Default islands

## Testing

Comprehensive tests are available in `tests/test_nuts.py`:
- Island creation and configuration
- Island model management
- Integration with NutsTheorists
- Backwards compatibility
- Performance tracking

Run tests with:
```bash
pytest tests/test_nuts.py -v
```
