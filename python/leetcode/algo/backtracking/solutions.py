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

    # 52. N-Queens II
    def totalNQueens(self, n: int) -> int:
        cols = set()
        diags1 = set()
        diags2 = set()
        ans = 0

        def is_safe(col, row):
            if col in cols:
                return False
            diag1 = row + col
            diag2 = row - col
            if diag1 in diags1 or diag2 in diags2:
                return False
            return True

        def backtrack(row):
            if row == n:
                nonlocal ans
                ans += 1
                return

            for col in range(n):

                if not is_safe(col, row):
                    continue
                cols.add(col)
                diags1.add(row + col)
                diags2.add(row - col)

                backtrack(row + 1)

                cols.remove(col)
                diags1.remove(row + col)
                diags2.remove(row - col)

        backtrack(0)
        return ans
