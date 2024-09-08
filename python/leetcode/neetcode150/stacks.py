from ast import List
from typing import Optional
from collections import defaultdict


class Solution:

    def evalRPN(self, tokens: List[str]) -> int:
        if len(tokens) == 1:
            return int(tokens[0])

        def operate(num1, num2, operator):
            ans = 0
            if operator == "+":
                ans = num1 + num2
            elif operator == "-":
                ans = num1 - num2
            elif operator == "*":
                ans = num1 * num2
            elif operator == "/":
                ans = int(num1 / num2)
            return ans

        stack = [tokens.pop(0), tokens.pop(0)]
        operators = {"+", "-", "*", "/"}

        while len(stack) > 1 or tokens:
            curr = tokens.pop(0)
            if curr in operators:
                num2 = int(stack.pop())
                num1 = int(stack.pop())
                ans = operate(num1, num2, curr)
                stack.append(ans)
            else:
                stack.append(curr)

        return stack.pop()
