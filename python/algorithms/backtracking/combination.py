from collections import defaultdict, deque
from typing import List, Tuple, Optional


class Combination:

    def all_combinations(start: int, end: int, size: int):
        result = []

        def backtrack(idx: int, current: list):
            if len(current) == size:
                result.append(current[:])
                return
            for i in range(idx, end + 1):
                current.append(i)
                backtrack(i + 1, current)
                current.pop()

        backtrack(0, [])
        return result
