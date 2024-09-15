from ast import List
from typing import Optional


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
