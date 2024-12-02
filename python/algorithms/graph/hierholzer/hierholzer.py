class Hierholzer:
    """
    - graph traversal algo to find Eulerian path/circuit (path visits every edge once, and cicuit visits and back to start).

    - Eulerian path rules for directed graph
        - each node's outdegree == indegree
        - or exactly 1 node has 1 more outdegree than its indegree, and another 1 node with 1 more indegree (2 odd degrees)

    - Eulerian Circuit
        - all even degrees

    - [start1, end2] -> [start3, end3] -> [start4, end4]

    - Steps
        1. check conditions - in/out degrees
        2. for circuit, start at any node with non-zero degree / for path, choose one with an odd degree
        3. construct path/circuit using dfs/stack
        4. reverse the order

    """

    def hierholzer(graph, start_vertex):
        stack = [start_vertex]
        circuit = []

        while stack:
            current = stack[-1]
            if graph[current]:  # If there are unused edges
                next_vertex = graph[current].pop()
                # Remove the edge in both directions (for undirected graph)
                graph[next_vertex].remove(current)
                stack.append(next_vertex)
            else:
                circuit.append(stack.pop())

        return circuit[::-1]
