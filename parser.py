from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor, Node
from parsimonious import exceptions
from pprint import pprint
from copy import deepcopy

from nodes import VarNode, ApplNode, ConstNode, TupleNode, \
     QuantifierNode, generate_mappings, str_mappings, \
     repr_mappings, fixity
from definitions import Fixity, Associativity

grammar_debug = False  # whether to print debug statements

# Sets
set_types = [('Set', r'Set', 'Set')]

set_consts = [('EmptySet', r'\emptyset', '\u2205')]

set_unary_ops = []

set_binary_ops = [[('SetDifference', r'\setminus', ' \\ ', Associativity.LEFT)],
                  [('Union', r'\cup', '\u222a', Associativity.LEFT), 
                   ('Intersection', r'\cap', '\u2229', Associativity.LEFT)],
                  [('CartesianProduct', r'\times', '\u00d7', Associativity.LEFT)]]

set_unary_fns = [('PowerSet', r'\mathcal{P}', '\u2118', Fixity.PREFIX, 1),
                 ('Complement', r'\complement', 'complement', Fixity.PREFIX, 1)]

set_binary_fns = []

set_unary_preds = []

set_binary_preds = [('Subset', r'\subset', ' \u2282 ', Fixity.INFIX),
                    ('SubsetEq', r'\subseteq', ' \u2286 ', Fixity.INFIX),
                    ('Element', r'\in', ' \u2208 ', Fixity.INFIX)]
                  
# Numbers
num_types = [('Natural', r'\\mathbb{N}', '\u2115'),
             ('Integer', r'\\mathbb{Z}', '\u2124'),
             ('Rational', r'\\mathbb{Q}', '\u211a'),
             ('Real', r'\\mathbb{R}', '\u211d')]

num_consts = []

num_unary_ops = [[('Negation', '-', '-', Associativity.NON)]]

num_binary_ops = [[('Addition', '+', ' + ', Associativity.LEFT),
                   ('Subtraction', '-', ' - ', Associativity.LEFT)],
                  [('Multiplication', '*', ' * ', Associativity.LEFT),
                   ('Division', '/', ' / ', Associativity.LEFT)],
                  [('Exponentiation', '^', ' ^ ', Associativity.RIGHT)]]

num_unary_fns = []

num_binary_fns = []

num_unary_preds = []

num_binary_preds = [('LessThan', '<', ' < ', Fixity.INFIX),
                    ('LessThanOrEqual', '<=', ' <= ', Fixity.INFIX),
                    ('Equal', '=', ' = ', Fixity.INFIX),
                    ('NotEqual', r'\neq', ' \u2260 ', Fixity.INFIX)]

# Groups
group_types = [('Group', r'Group', 'Group')]

group_consts = []

group_unary_ops = []

group_binary_ops = [[('Multiplication', '*', ' * ', Associativity.LEFT),
                   ('Division', '/', ' / ', Associativity.LEFT)],
                  [('Exponentiation', '^', ' ^ ', Associativity.RIGHT)]]

group_unary_fns = []

group_binary_fns = []

group_unary_preds = []

group_binary_preds = [('Equal', '=', ' = ', Fixity.INFIX),
                    ('NotEqual', r'\neq', ' \u2260 ', Fixity.INFIX)]
                    
class Operator:
    def __init__(self, name, notation, symbol, associativity, precedence):
        self.name = name
        self.notation = notation
        self.symbol = symbol
        self.associativity = associativity
        self.precedence = precedence

class Predicate:
    def __init__(self, name, notation, symbol, fixity, precedence):
        self.name = name
        self.notation = notation
        self.symbol = symbol
        self.fixity = fixity
        self.precedence = precedence

class Function:
    def __init__(self, name, notation, symbol, fixity, arity):
        self.name = name
        self.notation = notation
        self.symbol = symbol
        self.fixity = fixity
        self.arity = arity

def create_specification(set_types, set_consts, set_unary_ops, set_binary_ops, set_unary_fns, set_binary_fns, set_unary_preds, set_binary_preds):
    operators = []
    predicates = []
    functions = []
    constants = [(f"{const[0].lower()}_constant", const[1], const[2]) for const in set_consts]
    types = set_types

    def add_operators(op_list):
        for precedence, ops in enumerate(op_list):
            for op in ops:
                name, notation, symbol, associativity = op
                operators.append(Operator(name, notation, symbol, associativity, precedence))
        if grammar_debug:
            print("Operators added:", operators)

    def add_predicates(pred_list):
        for pred in pred_list:
            name, notation, symbol, pred_fixity = pred
            predicates.append(Predicate(name, notation, symbol, pred_fixity, precedence=0))  # Predicates don't have precedence levels in this setup
        if grammar_debug:
            print("Predicates added:", predicates)

    def add_functions(fn_list):
        for fn in fn_list:
            name, notation, symbol, fn_fixity, arity = fn
            functions.append(Function(name, notation, symbol, fn_fixity, arity))
        if grammar_debug:
            print("Functions added:", functions)

    add_operators(set_unary_ops)
    add_operators(set_binary_ops)
    add_functions(set_unary_fns)
    add_functions(set_binary_fns)
    add_predicates(set_unary_preds)
    add_predicates(set_binary_preds)

    if grammar_debug:
        print("Constants added:", constants)
        print("Types added:", types)

    return operators, predicates, functions, constants, types

def merge_specifications(*specs):
    merged_operators = {}
    merged_predicates = {}
    merged_functions = {}
    merged_constants = set()
    merged_types = set()
    precedence_groups = {}

    for spec in specs:
        operators, predicates, functions, constants, types = spec
        for operator in operators:
            key = (operator.name, operator.notation)
            if key not in merged_operators:
                merged_operators[key] = operator
                if (operator.precedence, operator.associativity) not in precedence_groups:
                    precedence_groups[(operator.precedence, operator.associativity)] = []
                precedence_groups[(operator.precedence, operator.associativity)].append(operator)
            else:
                existing_op = merged_operators[key]
                if existing_op.associativity != operator.associativity:
                    print(f"Conflict detected for operator: {operator.name}")
                    print(f"Existing operator: {existing_op.name}, {existing_op.notation}, {existing_op.precedence}, {existing_op.associativity}")
                    print(f"New operator: {operator.name}, {operator.notation}, {operator.precedence}, {operator.associativity}")
                    raise ValueError(f"Inconsistent associativity for operator: {operator.name}")

        for predicate in predicates:
            key = (predicate.name, predicate.notation)
            if key not in merged_predicates:
                merged_predicates[key] = predicate
            else:
                existing_pred = merged_predicates[key]
                if existing_pred.notation != predicate.notation:
                    raise ValueError(f"Inconsistent notation detected for predicate: {predicate.name}")

        for function in functions:
            key = (function.name, function.notation)
            if key not in merged_functions:
                merged_functions[key] = function
            else:
                existing_fn = merged_functions[key]
                if existing_fn.notation != function.notation:
                    raise ValueError(f"Inconsistent notation detected for function: {function.name}")

        merged_constants.update(constants)
        merged_types.update(types)

    # Separate precedence levels while ensuring order
    new_precedence = 0
    precedence_mapping = {}
    sorted_groups = sorted(precedence_groups.items(), key=lambda x: (x[0][0], x[0][1].name))
    for (orig_precedence, associativity), ops in sorted_groups:
        if (orig_precedence, associativity) not in precedence_mapping:
            precedence_mapping[(orig_precedence, associativity)] = new_precedence
            new_precedence += 1
        else:
            new_precedence = precedence_mapping[(orig_precedence, associativity)]

    sorted_operators = []  # Initialize the sorted_operators list
    for (orig_precedence, associativity), ops in sorted_groups:
        new_precedence = precedence_mapping[(orig_precedence, associativity)]
        for op in ops:
            sorted_operators.append(Operator(op.name, op.notation, op.symbol, op.associativity, new_precedence))

    sorted_predicates = list(merged_predicates.values())
    sorted_functions = list(merged_functions.values())
    sorted_constants = list(merged_constants)
    sorted_types = list(merged_types)

    if grammar_debug:
        print("Merged operators:", sorted_operators)
        print("Merged predicates:", sorted_predicates)
        print("Merged functions:", sorted_functions)
        print("Merged constants:", sorted_constants)
        print("Merged types:", sorted_types)

    return sorted_operators, sorted_predicates, sorted_functions, sorted_constants, sorted_types

def generate_parsimonious_grammar(operators, predicates, functions, constants, types):
    formula_definitions = []
    formula_rules = []
    term_definitions = []
    term_rules = []
    atomic_definitions = []

    def precedence_to_rule_name(precedence, kind):
        return f"{kind}_{precedence}"

    # Generate logical connectives and constants rules
    logical_rules = [
        r'logical_neg = "\\neg" logical_atomic',
        r'logical_iff = logical_implies space "\\iff" space logical_implies',
        r'logical_implies = logical_binary (space "\\implies" space logical_binary)*',
        r'logical_binary = logical_atomic (space ("\\wedge" / "\\vee") space logical_atomic)*',
        r'logical_atomic = logical_top / logical_bot / atomic_formula / logical_paren',
        r'logical_paren = "(" formula ")"',
        r'logical_top = "\\top"',
        r'logical_bot = "\\bot"'
    ]

    # Generate predicate rules (logical formulas)
    for predicate in predicates:
        formula_rule_name = f"{predicate.name.lower()}_formula"
        if predicate.fixity == Fixity.INFIX:
            formula_definitions.append(f"{formula_rule_name} = term space \"{predicate.notation}\" space term")
        elif predicate.fixity == Fixity.PREFIX:
            formula_definitions.append(f"{formula_rule_name} = \"{predicate.notation}\" space term")
        formula_rules.append(formula_rule_name)

    # Generate term rules with proper handling of left, right, and non-associativity
    precedence_levels = sorted(set(op.precedence for op in operators))
    for precedence in precedence_levels:
        rule_name = precedence_to_rule_name(precedence, 'term')
        next_rule = precedence_to_rule_name(precedence + 1, 'term') if precedence + 1 in precedence_levels else "atomic"

        ops_at_level = [op for op in operators if op.precedence == precedence]
        associativity_set = {op.associativity for op in ops_at_level}
        if len(associativity_set) > 1:
            print(f"Inconsistent associativity detected at precedence level {precedence}")
            print(f"Operators with conflicting associativity: {ops_at_level}")
            raise ValueError(f"Inconsistent associativity detected at precedence level {precedence}")

        associativity = associativity_set.pop()
        op_choices = " / ".join([f'"{op.notation}"' for op in ops_at_level])

        if associativity == Associativity.LEFT:
            term_definitions.append(f"{rule_name} = {next_rule} (space ({op_choices}) space {next_rule})*")
        elif associativity == Associativity.RIGHT:
            term_definitions.append(f"{rule_name} = ({next_rule} space ({op_choices}) space)* {next_rule}")
        elif associativity == Associativity.NON:
            term_definitions.append(f"{rule_name} = {next_rule} space ({op_choices}) space {next_rule}")
        else:
            raise ValueError(f"Unsupported associativity: {associativity}")

        term_rules.append(rule_name)

    # Generate function rules
    for function in functions:
        function_rule_name = f"{function.name.lower()}_fn"
        term_list = ', '.join(['term'] * function.arity)
        atomic_definitions.append(f"{function_rule_name} = \"{function.notation}\" \"(\" space {term_list} space \")\"")
        term_rules.append(function_rule_name)

    # User-defined function rules
    atomic_definitions.append("user_function = variable \"(\" space term (space \",\" space term)* space \")\"")
    term_rules.append("user_function")

    # Generate constant rules
    constant_rules = [f"{const[0]} = \"{const[1]}\"" for const in constants]

    # Append terminals
    atomic_definitions.append(f"""
atomic = variable / integer / {' / '.join([const[0] for const in constants])} / paren_formula / user_function / {' / '.join([f'{function.name.lower()}_fn' for function in functions])}
""")

    term_definitions.append(f"term = {' / '.join(term_rules)}")

    # Rename the original formula rule to logical_formula
    formula_definitions.append(f"atomic_formula = {' / '.join(formula_rules)}")

    # Add quantifier rules
    quantifier_rules = [
        r'existential_formula = "\\exists" space variable (space optional_type)? space formula',
        r'universal_formula = "\\forall" space variable (space optional_type)? space formula'
    ]

    # Add optional type specification rule
    type_names = " / ".join([f'"{type[1]}"' for type in types])
    type_specification_rules = [
        r'optional_type = (":" space type_name) / ("\\in" space term)',
        f'type_name = {type_names}'
    ]

    # Add logical_formula rule
    logical_formula_rule = """
logical_formula = logical_neg / logical_iff / logical_implies
"""

    # Ensure 'formula' is the first rule
    top_formula_rule = """
formula = existential_formula / universal_formula / logical_formula
"""

    atomic_definitions.append(r"""
variable = ~"[a-zA-Z_][a-zA-Z0-9_]*"
integer = ~"[0-9]+"
paren_formula = "(" space formula space ")"
space = ~"\s*"
""")

    # Assemble the grammar parts
    grammar_parts = [top_formula_rule] + quantifier_rules + type_specification_rules + [logical_formula_rule] + logical_rules + formula_definitions + term_definitions + constant_rules + atomic_definitions

    grammar = "\n".join(grammar_parts).strip()
    if grammar_debug:
        print("Generated grammar:")
        print(grammar)
    return grammar

class MathNodeVisitor(NodeVisitor):
    def visit_variable(self, node, visited_children):
        return VarNode(node.text)

    def visit_atomic(self, node, visited_children):
        return visited_children[0]

    def visit_term_4(self, node, visited_children):
        print(f"visit_term_4: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term_4: child {i} = {child}, type = {type(child)}")

        res = visited_children[1]
        if visited_children[0]:
            for operation in visited_children[0]:
                op_text = operation[1][0].text.strip()
                op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                if op_key:
                    op = ConstNode(op_key)
                    res = ApplNode(op, TupleNode((operation[3], res)))
                else:
                    print(f"Unknown operator: {op_text}")

        print(f"visit_term_4: resulting term = {res}")
        return res

    def visit_term_3(self, node, visited_children):
        print(f"visit_term_3: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term_3: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                op_text = operation[1][0].text.strip()
                op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                if op_key:
                    op = ConstNode(op_key)
                    res = ApplNode(op, TupleNode((res, operation[3])))
                else:
                    print(f"Unknown operator: {op_text}")

        print(f"visit_term_3: resulting term = {res}")
        return res

    def visit_term_2(self, node, visited_children):
        print(f"visit_term_2: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term_2: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                print(f"visit_term_2: operation = {operation}, type = {type(operation)}")
                if isinstance(operation, list) and len(operation) == 4:
                    print(f"visit_term_2: operation details = {operation}")
                    op_text = operation[1][0].text.strip()
                    op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                    if op_key:
                        op = ConstNode(op_key)
                        res = ApplNode(op, TupleNode((res, operation[3])))
                    else:
                        print(f"Unknown operator: {op_text}")

        print(f"visit_term_2: resulting term = {res}")
        return res

    def visit_term_1(self, node, visited_children):
        print(f"visit_term_1: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term_1: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                op_text = operation[1][0].text.strip()
                op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                if op_key:
                    op = ConstNode(op_key)
                    res = ApplNode(op, TupleNode((res, operation[3])))
                else:
                    print(f"Unknown operator: {op_text}")

        print(f"visit_term_1: resulting term = {res}")
        return res

    def visit_term_0(self, node, visited_children):
        print(f"visit_term_0: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term_0: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                op_text = operation[1][0].text.strip()
                op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                if op_key:
                    op = ConstNode(op_key)
                    res = ApplNode(op, TupleNode((res, operation[3])))
                else:
                    print(f"Unknown operator: {op_text}")

        print(f"visit_term_0: resulting term = {res}")
        return res

    def visit_term(self, node, visited_children):
        print(f"visit_term: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_term: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if len(visited_children) > 1 and visited_children[1]:
            for operation in visited_children[1]:
                op_text = operation[1][0].text.strip()
                op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                if op_key:
                    op = ConstNode(op_key)
                    res = ApplNode(op, TupleNode((res, operation[3])))
                else:
                    print(f"Unknown operator: {op_text}")

        print(f"visit_term: resulting term = {res}")
        return res

    def visit_element_formula(self, node, visited_children):
        print(f"visit_element_formula: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_element_formula: child {i} = {child}, type = {type(child)}")

        left = visited_children[0]
        right = visited_children[4]
        op_text = visited_children[2].text.strip()
        op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
        if op_key:
            op = ConstNode(op_key)
            element_node = ApplNode(op, TupleNode((left, right)))
        else:
            print(f"Unknown operator: {op_text}")

        print(f"visit_element_formula: {element_node}")
        return element_node

    def visit_atomic_formula(self, node, visited_children):
        return visited_children[0]

    def visit_logical_implies(self, node, visited_children):
        print(f"visit_logical_implies: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_logical_implies: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                print(f"visit_logical_implies: operation = {operation}, type = {type(operation)}")
                if isinstance(operation, list) and len(operation) == 4:
                    print(f"visit_logical_implies: operation details = {operation}")
                    op_text = operation[1][0].text.strip()
                    op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                    if op_key:
                        op = ConstNode(op_key)
                        res = ApplNode(op, TupleNode((res, operation[3])))
                    else:
                        print(f"Unknown operator: {op_text}")

        print(f"visit_logical_implies: resulting term = {res}")
        return res

    def visit_logical_binary(self, node, visited_children):
        print(f"visit_logical_binary: number of children = {len(visited_children)}")
        for i, child in enumerate(visited_children):
            print(f"visit_logical_binary: child {i} = {child}, type = {type(child)}")

        res = visited_children[0]
        if visited_children[1]:
            for operation in visited_children[1]:
                print(f"visit_logical_binary: operation = {operation}, type = {type(operation)}")
                if isinstance(operation, list) and len(operation) == 4:
                    print(f"visit_logical_binary: operation details = {operation}")
                    op_text = operation[1][0].text.strip()
                    op_key = next((k for k, v in repr_mappings.items() if v == op_text), None)
                    if op_key:
                        op = ConstNode(op_key)
                        res = ApplNode(op, TupleNode((res, operation[3])))
                    else:
                        print(f"Unknown operator: {op_text}")

        print(f"visit_logical_binary: resulting term = {res}")
        return res

    def visit_logical_atomic(self, node, visited_children):
        return visited_children[0]

    def visit_logical_formula(self, node, visited_children):
        return visited_children[0]

    def visit_formula(self, node, visited_children):
        return visited_children[0]

    def visit_space(self, node, visited_children):
        return None

    def generic_visit(self, node, visited_children):
        return visited_children or node

# Example usage
set_spec = create_specification(set_types, set_consts, set_unary_ops, set_binary_ops, set_unary_fns, set_binary_fns, set_unary_preds, set_binary_preds)
num_spec = create_specification(num_types, num_consts, num_unary_ops, num_binary_ops, num_unary_fns, num_binary_fns, num_unary_preds, num_binary_preds)
group_spec = create_specification(group_types, group_consts, group_unary_ops, group_binary_ops, group_unary_fns, group_binary_fns, group_unary_preds, group_binary_preds)

# Merge specifications
merged_spec = merge_specifications(set_spec, num_spec, group_spec)

# Generate the grammar
operators, predicates, functions, constants, types = merged_spec
grammar = generate_parsimonious_grammar(operators, predicates, functions, constants, types)

# Create the Grammar object
formula = Grammar(grammar)

# Generate mappings
generate_mappings(operators, predicates, functions, constants)

# Create the visitor instance
visitor = MathNodeVisitor()

# Example usage with parsing
#example_formula = r'\forall x : \mathbb{N} (x \in z \implies (\exists y (y = x \cup \emptyset)))'
example_formula = r'x \in z \cup w'
parsed_tree = formula.parse(example_formula)
result = visitor.visit(parsed_tree)

# Print the results using repr and str
print(repr(result))
print(str(result))