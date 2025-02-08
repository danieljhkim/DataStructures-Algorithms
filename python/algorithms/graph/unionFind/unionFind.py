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


class SimpleUnionFind:

    def __init__(self):
        self.parent = {}

    def find(self, x):  # find the root of x
        if x not in self.parent:
            self.parent[x] = x  # if not found, set x as root
        else:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # compression magic
        return self.parent[x]

    def union(self, x, y):  # connect x to y
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY
