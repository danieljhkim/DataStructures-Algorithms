from typing import Optional, List
import heapq

"""Classic Examples of Greedy Algorithms

Interval Scheduling:
- Problem: Select the maximum number of non-overlapping intervals.
- Greedy Rule: Always select the interval that finishes earliest.

Huffman Encoding:
- Problem: Build an optimal binary prefix code.
- Greedy Rule: Merge the two nodes with the smallest frequencies.

Kruskal’s Algorithm (for MST):
- Problem: Find the Minimum Spanning Tree.
- Greedy Rule: Always choose the smallest edge that doesn’t form a cycle.

Dijkstra’s Algorithm:
- Problem: Find the shortest path from a source to all other nodes.
- Greedy Rule: Expand the nearest unvisited node.

Activity Selection:
- Problem: Choose the maximum number of activities that don’t overlap.
- Greedy Rule: Pick the activity that ends earliest

Algos
- Prim's algorithm
- Kruskal's algorithm
- Huffman Coding
"""


class Solution:

    def max_occupancy(N: int, taken: list, gaps: int):
        """_summary_
        0 0 0 1 0 0 0 1 0 0
        [    ] [     ] [  ]
        """
        if not taken:
            return (N + gaps) // (gaps + 1)
        taken.sort()
        ans = 0
        # 0 to first
        first_gap = taken[0]
        if first_gap > gaps:
            ans += (first_gap) // (gaps + 1)

        for i in range(1, len(taken)):
            prev = taken[i - 1]
            next = taken[i]
            gap = next - prev - 1
            if gap > gaps:
                ans += gap // (gaps + 1)

        last_gap = N - taken[-1] - 1
        if last_gap > gaps:
            ans += last_gap // (gaps + 1)
        return ans

    def maxSubArray(self, nums: List[int]) -> int:
        """_summary_
        max sum of subarray
        """
        sub_arr_sum = nums[0]
        max_sum = sub_arr_sum
        for n in nums[1:]:
            sub_arr_sum = max(n, sub_arr_sum + n)
            max_sum = max(max_sum, sub_arr_sum)
        return max_sum

    def canJump(self, nums: List[int]) -> bool:
        """_summary_
        is it possible to make it to the end of the arr?
        """
        jumps = nums[0]
        if jumps <= 0 and len(nums) > 1:
            return False
        for i, n in enumerate(nums[1:]):
            jumps -= 1
            if jumps < 0:
                return False
            jumps = max(jumps, n)
        return True

    def jump(self, nums: List[int]) -> int:
        """_summary_
        - wrong solution
        min jumps to make it across
        """
        if len(nums) < 2:
            return 0
        cur_jump = nums[0]
        top_jump = nums[0]
        count = 1
        for n in nums[1:]:
            top_jump -= 1
            cur_jump -= 1
            top_jump = max(top_jump, n)
            if cur_jump < 0:
                count += 1
                cur_jump = top_jump
        if cur_jump == 0:
            count += 1
        return count
