"""348. Design Tic-Tac-Toe
Assume the following rules are for the tic-tac-toe game on an n x n board between two players:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves are allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
Implement the TicTacToe class:

TicTacToe(int n) Initializes the object the size of the board n.
int move(int row, int col, int player) Indicates that the player with id player plays at the cell (row, col) of the board. The move is guaranteed to be a valid move, and the two players alternate in making moves. Return
0 if there is no winner after the move,
1 if player 1 is the winner after the move, or
2 if player 2 is the winner after the move.
"""


# 348. Design Tic-Tac-Toe
class TicTacToe:

    def __init__(self, n: int):
        self.rows = [[0, 0] for _ in range(n)]
        self.cols = [[0, 0] for _ in range(n)]
        self.diag1, self.diag2 = [0, 0], [0, 0]
        self.n = n
        odd = n % 2 == 1

    def update(self, player, board):
        who = board[0]
        if who == 0 or who == player:
            board[0] = player
            board[1] += 1
            if board[1] == self.n:
                return 1
        elif who != player:
            board.clear()
        return 0

    def move(self, row: int, col: int, player: int) -> int:
        if row + col == self.n - 1 and self.diag2:
            res = self.update(player, self.diag2)
            if res:
                return player
        if row == col and self.diag1:
            res = self.update(player, self.diag1)
            if res:
                return player

        if self.rows[row]:
            res = self.update(player, self.rows[row])
            if res:
                return player
        if self.cols[col]:
            res = self.update(player, self.cols[col])
            if res:
                return player
        return 0
