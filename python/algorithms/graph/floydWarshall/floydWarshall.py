"""Floyd Warshall Algorithm
    - shortest paths between all paris of nodes
"""


def floyd_warshall(graph):
    n = len(graph)
    distances = [[float("infinity") for _ in range(n)] for _ in range(n)]

    for i in range(n):
        distances[i][i] = 0
        # we set start position to 0 for each node

    for node in graph:
        for neighbor, weight in graph[node].items():
            distances[node][neighbor] = weight

    for middle in range(n):
        for src in range(n):
            for dest in range(n):
                # is src -> middle -> dest shorter than src -> dest?
                distances[src][dest] = min(
                    distances[src][middle],
                    distances[src][middle] + distances[middle][dest],
                )

    return distances
