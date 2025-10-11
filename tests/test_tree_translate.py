import unittest
from autora.theorist.nuts import NutsTheorists  # Adjust if your structure is different

class TestTreeTranslate(unittest.TestCase):
    def setUp(self):
        self.model = NutsTheorists()

    def test_long_nested_expression_1(self):
        tree = ['+', 
                    ['np.exp', ['*', 'S1', 'c']], 
                    ['/', 
                        ['np.power', ['-', 'S2', 'c'], 'c'], 
                        ['np.log', ['+', 'S1', 'S2']]
                    ]
                ]
        expected = '(exp((S1 * c)) + (((S2 - c) ^ c) / log((S1 + S2))))'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_long_nested_expression_2(self):
        tree = ['np.power', 
                    ['+', 
                        ['np.log', ['*', 'S1', 'S2']], 
                        ['np.exp', 'c']
                    ],
                    ['-', 'S1', ['/', 'S2', 'c']]
                ]
        expected = '((log((S1 * S2)) + exp(c)) ^ (S1 - (S2 / c)))'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_log_expression(self):
        tree = ['np.log', ['*', 'S1', 'S2']]
        expected = 'log((S1 * S2))'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_exp_subtract(self):
        tree = ['-', ['np.exp', 'S2'], 'S1']
        expected = '(exp(S2) - S1)'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_power_expression(self):
        tree = ['np.power', 'S1', 'c']
        expected = '(S1 ^ c)'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_div_addition(self):
        tree = ['/', ['+', 'S1', 'S2'], 'c']
        expected = '((S1 + S2) / c)'
        self.assertEqual(self.model._tree_translate(tree), expected)

    def test_unknown_operator(self):
        with self.assertRaises(ValueError):
            self.model._tree_translate(['unknown_op', 'S1'])

if __name__ == '__main__':
    unittest.main()
