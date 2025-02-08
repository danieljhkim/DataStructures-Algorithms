# Dikstra Algorithm

---

- Shortest path algorithm
- Good for weighted graphs
- Not good for negative weights -> use Bellman-Ford or Floyd-Warshall instead
- It's really just a BFS with extra steps
- Very simple yet powerful
- O(E + V log V)

```python
def dikstra(graph, start, end):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        cur_dist, current_node = heapq.heappop(queue)

        if current_node == end: # reached destination, so return current distance
            return cur_dist

        for neighbor, weight in graph[current_node].items():
            distance = cur_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor)) # heap orders by distance -> shortest comes first

    return -1

```