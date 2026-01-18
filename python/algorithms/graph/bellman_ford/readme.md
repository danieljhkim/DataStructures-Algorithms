# Bellman-Ford Algorithm

---

- Shortest Path Algorithm, to find shortest paths from a single source to all other nodes
- Good for weighted graphs, both positive and negative weights
- Detects negative weight cycles
- O(V * E) time complexity

```python

def bellman_ford(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1): # why? because a node can have at most V - 1 edges
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    for node in graph:
        for neighbor, weight in graph[node].items():
            if distances[node] + weight < distances[neighbor]:
                # keeps getting smaller -> negative cycle
                return "Negative Cycle Detected"

    return distances

```