# Floyd-Warshall Algorithm

--- 

- Shortest paths to all other nodes from all nodes
- Good for weighted graphs
- O(V^3) time complexity

```python
def floyd_warshall(graph):
    n = len(graph)
    distances = [[float('infinity') for _ in range(n)] for _ in range(n)]
   
    """
    we will find:
    distances[i][j] = shortest path from i to j
    """

    for i in range(n):
        distances[i][i] = 0
        # we set start position to 0 for each node

    for node in graph:
        for neighbor, weight in graph[node].items():
            distances[node][neighbor] = weight # initial distances

    for middle in range(n):
        for src in range(n):
            for dest in range(n):
                # is src -> middle -> dest shorter than src -> dest?
                distances[src][dest] = min(distances[src][middle], distances[src][middle] + distances[middle][dest])

    return distances
```