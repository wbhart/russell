import unittest
from parser import formula, visitor
from nodes import VarNode, ApplNode, ConstNode, TupleNode

class TestMathParser(unittest.TestCase):

    def compare_trees(self, expected, result):
        """Helper function to compare expected and result parse trees."""
        self.assertEqual(repr(expected), repr(result))
        self.assertEqual(str(expected), str(result))

    def test_sets(self):
        test_cases = [
            {
                "input": r"x \in z \cup w",
                "expected": ApplNode(ConstNode('Element'), TupleNode((VarNode('x'), ApplNode(ConstNode('Union'), TupleNode((VarNode('z'), VarNode('w')))))))
            },
            {
                "input": r"a \subseteq b \cap c",
                "expected": ApplNode(ConstNode('SubsetEq'), TupleNode((VarNode('a'), ApplNode(ConstNode('Intersection'), TupleNode((VarNode('b'), VarNode('c')))))))
            }
        ]
        for case in test_cases:
            with self.subTest(case=case):
                parsed_tree = formula.parse(case["input"])
                result = visitor.visit(parsed_tree)
                self.compare_trees(case["expected"], result)

    def test_numbers(self):
        test_cases = [
            {
                "input": r"a = x + y * z",
                "expected": ApplNode(ConstNode('Equal'), TupleNode((VarNode('a'), ApplNode(ConstNode('Addition'), TupleNode((VarNode('x'), ApplNode(ConstNode('Multiplication'), TupleNode((VarNode('y'), VarNode('z'))))))))))
            },
            {
                "input": r"a / b - c \neq b + c",
                "expected": ApplNode(ConstNode('NotEqual'), TupleNode((ApplNode(ConstNode('Subtraction'), TupleNode((ApplNode(ConstNode('Division'), TupleNode((VarNode('a'), VarNode('b')))), VarNode('c')))), ApplNode(ConstNode('Addition'), TupleNode((VarNode('b'), VarNode('c')))))))
            }
        ]
        for case in test_cases:
            with self.subTest(case=case):
                parsed_tree = formula.parse(case["input"])
                result = visitor.visit(parsed_tree)
                self.compare_trees(case["expected"], result)

    def test_groups(self):
        test_cases = [
            {
                "input": r"h = g * h ^ k",
                "expected": ApplNode(ConstNode('Multiplication'), TupleNode((VarNode('g'), ApplNode(ConstNode('Exponentiation'), TupleNode((VarNode('h'), VarNode('k')))))))
            },
            {
                "input": r"g = a / b * c",
                "expected": ApplNode(ConstNode('Multiplication'), TupleNode((ApplNode(ConstNode('Division'), TupleNode((VarNode('a'), VarNode('b')))), VarNode('c'))))
            }
        ]
        for case in test_cases:
            with self.subTest(case=case):
                parsed_tree = formula.parse(case["input"])
                result = visitor.visit(parsed_tree)
                self.compare_trees(case["expected"], result)

if __name__ == '__main__':
    unittest.main()

