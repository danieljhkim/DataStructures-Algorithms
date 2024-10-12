from typing import Optional, List


class Solutions:

    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        left = 0
        ans = ""
        for c in s:
            if c == "(":
                left += 1
            elif c == ")":
                if left == 0:
                    continue
                left -= 1
            stack.append(c)
        while stack:
            c = stack.pop()
            if c == "(" and left > 0:
                left -= 1
                continue
            ans = c + ans
        return ans
