from typing import Optional, List
import bisect

"""3814. Maximum Capacity Within Budget
You are given two integer arrays costs and capacity, both of length n, where costs[i] represents the purchase cost of the ith machine and capacity[i] represents its performance capacity.

You are also given an integer budget.

You may select at most two distinct machines such that the total cost of the selected machines is strictly less than budget.

Return the maximum achievable total capacity of the selected machines.
"""


class Solution:
    """
    This was my intuitive solution. Not quite as optimal but made sense to me.
    """

    def maxCapacity(self, costs: List[int], capacity: List[int], budget: int) -> int:
        cap_arr, cost_arr = [], []
        for i in range(len(costs)):
            if costs[i] >= budget:
                continue
            cap_arr.append((capacity[i], costs[i], i))
            cost_arr.append((-costs[i], i, capacity[i]))
        if not cap_arr:
            return 0

        cap_arr.sort(reverse=True)
        cost_arr.sort()
        best = cap_arr[0][0]
        for val, cval, idx in cap_arr:
            if not cost_arr:
                break
            target = budget - cval - 1
            found = bisect.bisect_left(cost_arr, -target, key=lambda x: x[0])
            pop_count = len(cost_arr) - found
            for _ in range(pop_count):
                c, j, cp = cost_arr.pop()
                if j != idx:
                    best = max(best, val + cp)
        return best

    """
    Solution by townizm. 
    
    Similar idea, but more refined. 
    """

    def maxCapacity(self, costs: List[int], caps: List[int], budget: int) -> int:
        costs, caps = zip(*sorted([(c, cap) for c, cap in zip(costs, caps)]))
        res = 0
        stack: list[tuple[int, int]] = [(0, 0)]  # [(cost, cap)]
        for cost, cap in zip(costs, caps):
            while stack and stack[-1][0] + cost >= budget:
                stack.pop()
            if stack:
                res = max(res, stack[-1][1] + cap)
            if not stack or stack[-1][1] < cap:
                stack.append((cost, cap))
        return res
