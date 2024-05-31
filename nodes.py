from enum import Enum
from definitions import Fixity, Associativity

# Global mappings
str_mappings = {}
repr_mappings = {}
fixity = {}

class ParseNode:
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class VarNode(ParseNode):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

class ApplNode(ParseNode):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __repr__(self):
        fn_str = repr(self.fn)
        arg_strs = [repr(arg) for arg in self.args.args]
        if fixity[self.fn.name] == Fixity.INFIX.value:
            result = f" {fn_str} ".join(arg_strs)
        elif fixity[self.fn.name] == Fixity.PREFIX.value:
            result = f"{fn_str}({', '.join(arg_strs)})"
        elif fixity[self.fn.name] == Fixity.POSTFIX.value:
            result = f"({', '.join(arg_strs)}){fn_str}"
        else:
            result = f"{fn_str}({', '.join(arg_strs)})"
        return result

    def __str__(self):
        fn_str = str_mappings.get(self.fn.name, self.fn.name)
        arg_strs = [str(arg) for arg in self.args.args]
        if fixity[self.fn.name] == Fixity.INFIX.value:
            result = f" {fn_str} ".join(arg_strs)
        elif fixity[self.fn.name] == Fixity.PREFIX.value:
            result = f"{fn_str}({', '.join(arg_strs)})"
        elif fixity[self.fn.name] == Fixity.POSTFIX.value:
            result = f"({', '.join(arg_strs)}){fn_str}"
        else:
            result = f"{fn_str}({', '.join(arg_strs)})"
        return result

class ConstNode(ParseNode):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return repr_mappings[self.name]

    def __str__(self):
        return str_mappings[self.name]

class TupleNode(ParseNode):
    def __init__(self, args):
        self.args = args

    def __repr__(self):
        return f"({', '.join(repr(arg) for arg in self.args)})"

    def __str__(self):
        return f"({', '.join(str(arg) for arg in self.args)})"

class QuantifierNode(ParseNode):
    def __init__(self, quantifier_type, variable, formula):
        self.quantifier_type = quantifier_type
        self.variable = variable
        self.formula = formula

    def __repr__(self):
        quantifier_str = "\\forall" if self.quantifier_type == QuantifierType.FORALL else "\\exists"
        return f"{quantifier_str} {repr(self.variable)} {repr(self.formula)}"

    def __str__(self):
        quantifier_str = "∀" if self.quantifier_type == QuantifierType.FORALL else "∃"
        return f"{quantifier_str} {str(self.variable)} {str(self.formula)}"

def generate_mappings(operators, predicates, functions, constants):
    global str_mappings, repr_mappings, fixity

    for operator in operators:
        str_mappings[operator.name] = operator.symbol
        repr_mappings[operator.name] = operator.notation
        fixity[operator.name] = Fixity.INFIX.value if operator.associativity else operator.associativity.value

    for predicate in predicates:
        str_mappings[predicate.name] = predicate.symbol
        repr_mappings[predicate.name] = predicate.notation
        fixity[predicate.name] = predicate.fixity.value

    for function in functions:
        str_mappings[function.name] = function.symbol
        repr_mappings[function.name] = function.notation
        fixity[function.name] = function.fixity.value

    for const in constants:
        name, notation, symbol = const
        str_mappings[name] = symbol
        repr_mappings[name] = notation