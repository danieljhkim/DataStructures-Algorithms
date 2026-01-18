# Kahns Algorithm

- For Directed Graphs
- It can detect cycles
- It can find the valid ordering of nodes to visit
- When you need to find which courses to take, in a valid order, you can use this algorithm. 
- O(V + E) time complexity

```python

def topo_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return topo_order

```
