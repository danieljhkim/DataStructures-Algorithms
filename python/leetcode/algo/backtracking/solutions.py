from ast import List
from typing import Optional


class Solution:

    # 39. Combination Sum
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []

        def backtrack(idx, comb, total):
            if total == target:
                ans.append(comb.copy())
                return
            if idx >= len(candidates) or target < total:
                return
            comb.append(candidates[idx])
            backtrack(idx, comb, total + candidates[idx])
            comb.pop()
            backtrack(idx + 1, comb, total)

        backtrack(0, [], 0)
        return ans

    # 46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        perms = []

        def backtrack(perm, idx):
            if idx == len(nums):
                perms.append(perm[:])

            for i in range(idx, len(perm)):
                perm[i], perm[idx] = perm[idx], perm[i]
                backtrack(perm[:], idx + 1)
                perm[i], perm[idx] = perm[idx], perm[i]

        backtrack(nums, 0)
        return perms
