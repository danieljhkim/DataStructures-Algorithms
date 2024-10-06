from typing import Optional
from typing import List


class Solution:

    # 51. N-Queens
    def solveNQueens(self, n):

        def create_board(board):
            ans = []
            for row in board:
                ans.append("".join(row))
            return ans

        def backtrack(row, diagonals1, diagonals2, cols, board):
            if row == n:  # base case
                outcomes.append(create_board(board))
                return

            for col in range(n):
                diag1 = row + col
                diag2 = col - row
                if diag1 in diagonals1 or diag2 in diagonals2 or col in cols:
                    continue
                diagonals1.add(diag1)
                diagonals2.add(diag2)
                cols.add(col)
                board[row][col] = "Q"

                backtrack(row + 1, diagonals1, diagonals2, cols, board)

                diagonals1.remove(diag1)
                diagonals2.remove(diag2)
                cols.remove(col)
                board[row][col] = "."

        outcomes = []
        board = [["."] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), board)
        return outcomes

    # 78. Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)

        def backtrack(arr, start_pos, size):
            if start_pos == size:
                ans.append(arr[:])
                return
            for i in range(start_pos, n):
                arr.append(nums[i])
                backtrack(arr, i + 1, size)
                arr.pop()

        for size in range(n + 1):
            backtrack([], 0, size)

        return ans

    # 39. Combination Sum
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []

        def backtrack(arr, idx):
            total = sum(arr)
            if total == target:
                ans.append(arr[:])
                return
            elif total > target:
                return

            for i in range(idx, len(candidates)):
                arr.append(candidates[i])
                backtrack(arr, i)
                arr.pop()

        backtrack([], 0)
        return ans

    # 46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        ans = []

        def backtrack(arr, pos):
            if pos == len(arr):
                ans.append(arr[:])
                return
            for i in range(pos, len(nums)):
                arr[pos], arr[i] = arr[i], arr[pos]
                backtrack(arr, pos + 1)
                arr[pos], arr[i] = arr[i], arr[pos]

        backtrack(nums, 0)
        return ans

    # 90. Subsets II
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        num_set = set()
        ans = []

        def backtrack(arr, idx, size):
            if size == idx and tuple(sorted(arr)) not in num_set:
                num_set.add(tuple(sorted(arr)))
                ans.append(arr[:])
                return
            for i in range(idx, len(nums)):
                arr.append(nums[i])
                backtrack(arr, i + 1, size)
                arr.pop()

        for i in range(len(nums) + 1):
            backtrack([], 0, i)
        return ans


def test_solution():
    s = Solution()
    a = [1, 2, 3]
    print(s.permute(a))


if __name__ == "__main__":
    test_solution()
