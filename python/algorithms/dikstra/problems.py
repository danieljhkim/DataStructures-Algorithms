import heapq
from typing import Dict, List
from math import inf


class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        """_summary_
        - 0: empty
        - 1: building
        - 2: obstacle

        [0 0 0 1 0]
        [1 0 2 0 1]
        [0 0 1 0 0]
        """
