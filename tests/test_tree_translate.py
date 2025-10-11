import pytest
from autora.theorist.nuts import NutsTheorists

def test_long_nested_expression_1():
    model = NutsTheorists()
    tree = ['+', 
                ['np.exp', ['*', 'S1', 'c']], 
                ['/', 
                    ['np.power', ['-', 'S2', 'c'], 'c'], 
                    ['np.log', ['+', 'S1', 'S2']]
                ]
            ]
    expected = '(exp((S1 * c)) + (((S2 - c) ^ c) / log((S1 + S2))))'
    assert model._tree_translate(tree) == expected

def test_log_expression():
    model = NutsTheorists()
    tree = ['np.log', ['*', 'S1', 'S2']]
    expected = 'log((S1 * S2))'
    assert model._tree_translate(tree) == expected

def test_power_expression():
    model = NutsTheorists()
    tree = ['np.power', 'S1', 'c']
    expected = '(S1 ^ c)'
    assert model._tree_translate(tree) == expected
