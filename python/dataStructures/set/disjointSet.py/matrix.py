class UnionFind:

    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.n = self.rows * self.cols
        self.parent = list(range(self.n))
        self.rank = [1] * self.n

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

    def node_id(self, x, y):
        return x * self.cols + y
