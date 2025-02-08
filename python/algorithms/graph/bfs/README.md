# BFS
---
- Bread and butter of graph algorithms, in my opinion, along with DFS.
- It traverses a graph level by level, so it's really intuitive to visualize.
- Good for finding shortest paths for unweighted graphs. Shorest path + unweighted graph => BFS.
- Very versatile and can be used in many different problems.
- O(V + E) time complexity

```python

def bfs(graph, start, end):
    visited = set([start])
    queue = deque([(start, 0)])

    while queue:
        current, dist = queue.popleft()
        if current == end:
            return dist

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
```