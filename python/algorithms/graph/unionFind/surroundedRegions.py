from typing import Optional, List


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


class Solution:
    """_summary_
    - find regions in grid where it is surrounded by X and not touching the edges
    """

    def solve(self, board: List[List[str]]) -> None:
        """_summary_
        1. create union find map, and tie each O to a root.
            - ones touching the edges are the edge root.
            - and ones not tied to the edge root are not surrounded
        2. find O those that dont have edge root
        """

        def node_id(r, c):
            return r * COL + c

        ROW = len(board)
        COL = len(board[0])
        edge_root = ROW * COL
        union_find = UnionFind(edge_root + 1)  # last one is the edge root
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for r in range(ROW):
            for c in range(COL):
                if board[r][c] == "O":
                    if r in [0, ROW - 1] or c in [0, COL - 1]:
                        union_find.union(node_id(r, c), edge_root)
                    else:
                        for dr, dc in directions:
                            nr = r + dr
                            nc = c + dc
                            if 0 <= nr < ROW and 0 <= nc < COL and board[nr][nc] == "O":
                                union_find.union(node_id(r, c), node_id(nr, nc))

        for r in range(ROW):
            for c in range(COL):
                root = union_find.find(node_id(r, c))
                if board[r][c] == "O" and root != union_find.find(edge_root):
                    board[r][c] = "X"
