"""Kahn's algorithm
    - directed graph cycle detection
    - sorting by least dependencies
"""

from collections import defaultdict, deque


def topological_sort(n, edges):
    indegree = [0] * n
    adj = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)
        indegree[dst] += 1

    queue = deque()
    for i, n in enumerate(indegree):
        if n == 0:
            queue.append(i)

    topological_order = []
    while queue:
        cur = queue.popleft()
        topological_order.append(cur)
        for node in adj[cur]:
            indegree[node] -= 1
            if indegree[node] == 0:
                queue.append(node)
    if len(topological_order) != n:
        return "Cycle detected"

    return topological_order
