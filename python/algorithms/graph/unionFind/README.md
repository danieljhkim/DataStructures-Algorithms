# Union Find

- How to check if a node is connected to another node in a graph? 

- How to check what group an element is a part of?

Union Find is the way. I love this algorithm, no cap. It's so simple and elegant - simply bussin'. 


```python
class SimpleUnionFind:

    def __init__(self):
        self.parent = {}

    def find(self, x): # find the root of x
        if x not in self.parent:
            self.parent[x] = x # if not found, set x as root
        else:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x]) # compression magic
        return self.parent[x]

    def union(self, x, y): # connect x to y
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY
```

