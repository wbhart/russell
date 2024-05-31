from enum import Enum

class Fixity(Enum):
    PREFIX = "prefix"
    INFIX = "infix"
    POSTFIX = "postfix"

class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"
    NON = "non"
