import random
import pytest
import numpy as np
import pandas as pd
from autora.theorist.nuts import NutsTheorists, Island, IslandModel


@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    random.seed(180)
    np.random.seed(180)
    return


def test():
    theorist = NutsTheorists()
    assert theorist is not None


def test_island_creation():
    """Test that Island objects can be created with various parameters."""
    island = Island("TestIsland", population_size=100, mutation_rate=0.3, 
                   tournament_size=20, elitism=10)
    assert island.name == "TestIsland"
    assert island.population_size == 100
    assert island.mutation_rate == 0.3
    assert island.tournament_size == 20
    assert island.elitism == 10
    assert island.is_alive == True
    assert island.best_fitness == float('inf')


def test_island_operator_weights():
    """Test that island operator weights work correctly."""
    weights = {'unary': 2.0, 'binary': 1.0, 'crossover': 1.5, 'mutation': 0.5}
    island = Island("TestIsland", operator_weights=weights)
    assert island.get_operator_weight('unary') == 2.0
    assert island.get_operator_weight('binary') == 1.0
    assert island.get_operator_weight('crossover') == 1.5
    assert island.get_operator_weight('mutation') == 0.5
    assert island.get_operator_weight('nonexistent') == 1.0  # default


def test_island_model_creation():
    """Test that IslandModel can be created with islands."""
    islands = [
        Island("Island1", population_size=50),
        Island("Island2", population_size=50),
        Island("Island3", population_size=50),
    ]
    model = IslandModel(islands)
    assert len(model.islands) == 3
    assert model.global_best_fitness == float('inf')


def test_island_model_get_rankings():
    """Test that island rankings are correctly computed."""
    islands = [
        Island("Island1", population_size=50),
        Island("Island2", population_size=50),
        Island("Island3", population_size=50),
    ]
    islands[0].best_fitness = 0.5
    islands[1].best_fitness = 0.3
    islands[2].best_fitness = 0.7
    
    model = IslandModel(islands)
    rankings = model.get_island_rankings()
    
    assert len(rankings) == 3
    assert rankings[0]['name'] == "Island2"  # best fitness
    assert rankings[1]['name'] == "Island1"
    assert rankings[2]['name'] == "Island3"


def test_nuts_with_island_model_simple(seed):
    """Test NutsTheorists with island model on simple data."""
    # Simple linear data y = 2*x + 1
    X = pd.DataFrame({'x': np.linspace(0, 10, 20)})
    y = pd.DataFrame({'y': 2 * X['x'] + 1})
    
    # Create custom islands with small populations for quick test
    islands = [
        Island("Fast", population_size=20, mutation_rate=0.1, elitism=5),
        Island("Slow", population_size=20, mutation_rate=0.3, elitism=3),
    ]
    
    theorist = NutsTheorists(
        n_generation=3,  # Just 3 generations for quick test
        verbose=False,
        use_island_model=True,
        islands=islands,
        n_constants=1
    )
    
    theorist.fit(X, y)
    
    # Check that we got some result
    assert theorist.best_tree is not None
    assert theorist.best_fitness < float('inf')
    assert theorist.island_model is not None
    
    # Check that islands tracked their stats
    for island in islands:
        assert len(island.generation_history) > 0


def test_nuts_without_island_model(seed):
    """Test that NutsTheorists still works in standard mode."""
    X = pd.DataFrame({'x': np.linspace(0, 10, 20)})
    y = pd.DataFrame({'y': 2 * X['x'] + 1})
    
    theorist = NutsTheorists(
        population_size=50,
        n_generation=3,
        verbose=False,
        use_island_model=False
    )
    
    theorist.fit(X, y)
    
    assert theorist.best_tree is not None
    assert theorist.island_model is None


def test_default_islands_creation(seed):
    """Test that default islands are created when use_island_model=True but no islands provided."""
    X = pd.DataFrame({'x': np.linspace(0, 10, 20)})
    y = pd.DataFrame({'y': X['x'] ** 2})
    
    theorist = NutsTheorists(
        n_generation=2,
        verbose=False,
        use_island_model=True
    )
    
    theorist.fit(X, y)
    
    assert theorist.island_model is not None
    assert len(theorist.island_model.islands) == 4  # default has 4 islands
    assert theorist.best_tree is not None


def test_island_summary_methods():
    """Test that island summary methods work correctly."""
    islands = [
        Island("Island1", population_size=50),
        Island("Island2", population_size=50),
    ]
    islands[0].best_fitness = 0.5
    islands[1].best_fitness = 0.3
    
    theorist = NutsTheorists(use_island_model=True, islands=islands)
    theorist.island_model.global_best_fitness = 0.3
    theorist.island_model.global_best_island = "Island2"
    
    summary = theorist.get_island_summary()
    assert summary is not None
    assert 'rankings' in summary
    assert 'global_best_island' in summary
    assert summary['global_best_island'] == "Island2"


def test_island_model_kill_weak_islands():
    """Test that weak islands can be killed."""
    islands = [
        Island("Strong", population_size=50),
        Island("Weak", population_size=50),
    ]
    islands[0].best_fitness = 0.3
    islands[1].best_fitness = 10.0
    
    model = IslandModel(islands)
    model.kill_weak_islands(fitness_threshold=5.0)
    
    assert islands[0].is_alive == True
    assert islands[1].is_alive == False


def test_island_model_get_migration_candidates():
    """Test getting migration candidates from top islands."""
    islands = [
        Island("Island1", population_size=50),
        Island("Island2", population_size=50),
        Island("Island3", population_size=50),
    ]
    
    # Set up some best trees
    islands[0].best_fitness = 0.5
    islands[0].best_tree = ['x']
    islands[0].best_constants = {}
    
    islands[1].best_fitness = 0.3
    islands[1].best_tree = ['y']
    islands[1].best_constants = {}
    
    islands[2].best_fitness = 0.7
    islands[2].best_tree = ['z']
    islands[2].best_constants = {}
    
    model = IslandModel(islands)
    candidates = model.get_migration_candidates(n_islands=2, n_trees=5)
    
    assert len(candidates) <= 10  # at most 2 islands * 5 trees
    # First candidate should be from best island
    assert candidates[0][1] == 0.3  # best fitness


def test_nuts_predict_with_island_model(seed):
    """Test that predictions work after fitting with island model."""
    X_train = pd.DataFrame({'x': np.linspace(0, 5, 15)})
    y_train = pd.DataFrame({'y': 2 * X_train['x'] + 1})
    
    X_test = pd.DataFrame({'x': np.linspace(6, 10, 5)})
    
    theorist = NutsTheorists(
        n_generation=3,
        verbose=False,
        use_island_model=True,
        n_constants=1
    )
    
    theorist.fit(X_train, y_train)
    predictions = theorist.predict(X_test)
    
    assert predictions is not None
    assert len(predictions) == 5


# Note: We encourage you to adjust this test and write more tests.
