import enum
import math


class OperatorType(enum.Enum):
    Function = 2
    MathOperator = 1


class Operator:
    def __init__(self):
        self.operator_type = OperatorType.Function
        self.precedence = -1
        self.left_associative = False
        self.large_num = 2 ** 64 - 1

    def solve(self, input1, input2) -> float:
        return -1


class Plus(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.MathOperator
        self.precedence = 2
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        return input1 + input2


class Minus(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.MathOperator
        self.precedence = 2
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        return input1 - input2


class Divide(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.MathOperator
        self.precedence = 3
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        if input2 == 0:
            return self.large_num if input1 >= 0 else -1 * self.large_num
        return float(float(input1) / float(input2))


class Multiply(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.MathOperator
        self.precedence = 3
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        return input1 * input2


class Power(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.MathOperator
        self.precedence = 4
        self.left_associative = False

    def solve(self, input1, input2) -> float:
        if input1 <= -2 and input2 >= 100:
            if input2 % 2 == 0:
                return self.large_num
            return -self.large_num
        if input1 >= 2 and input2 >= 100:
            return self.large_num
        if input1 < 0 and 1 > input2 > -1 and input2 != 0:
            return float('-inf')
        return math.pow(input1, input2)


class Sinus(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.Function
        self.precedence = 0
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        if input1 is None:
            return math.sin(input2)
        return math.sin(input1)


class Cosinus(Operator):
    def __init__(self):
        super().__init__()
        self.operator_type = OperatorType.Function
        self.precedence = 0
        self.left_associative = True

    def solve(self, input1, input2) -> float:
        if input1 is None:
            return math.cos(input2)
        return math.cos(input1)


class MathHelper:
    def __init__(self):
        self.operations_dict = {}
        self.operations_str = {"+": " + ", "-": " - ", "*": " * ", "/": " / ", "^": " ^ ", "S": "sin", "C": "cos"}

    def get_operations_dict(self):
        if len(self.operations_dict) == 0:
            self.make_operations()

        return self.operations_dict

    def make_operations(self):
        plus_o = Plus()
        minus_o = Minus()
        multiply_o = Multiply()
        divide_o = Divide()
        power_o = Power()
        sin_o = Sinus()
        cos_o = Cosinus()
        self.operations_dict = {"+": plus_o, "-": minus_o, "*": multiply_o, "/": divide_o, "^": power_o,
                                "S": sin_o, "C": cos_o}
