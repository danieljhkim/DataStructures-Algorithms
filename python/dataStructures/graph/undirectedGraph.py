from collections import defaultdict
from types import List

"""
Graph
    Definition: A graph is a collection of nodes (vertices) connected by edges.
    
    Structure: Consists of vertices and edges, where edges can connect any pair of vertices.
    
    Properties:
        - Cyclic or Acyclic: Can have cycles (cyclic) or no cycles (acyclic).
        - Connected or Disconnected: Can be connected (one path between any two vertices) or disconnected (some vertices are not reachable from others).
        - Directed or Undirected: Edges can have a direction (directed graph) or no direction (undirected graph).
        - Weighted or Unweighted: Edges can have weights (weighted graph) or no weights (unweighted graph).
"""


class UndirectedGraph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)  # For undirected graph

    def countConnectedComponents(self, n: int, edges: List[List[int]]) -> int:
        table = defaultdict(set)
        for edge in edges:
            table[edge[0]].add(edge[1])
            table[edge[1]].add(edge[0])
        visited = set()

        def dfs(node, group):
            if node in group:
                return
            cons = table[node]
            group.add(node)
            for c in cons:
                if c not in group:
                    dfs(c, group)
            return group

        ans = 0
        for key in table.keys():
            if key not in visited:
                group = dfs(key, set())
                ans += 1
                visited.update(group)
        if n > len(visited):
            alone = n - len(visited)
            ans += alone
        return ans


def print_graph(graph):
    for node in graph.adj_list:
        print(f"{node}: {graph.adj_list[node]}")


if __name__ == "main":
    graph = UndirectedGraph()
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 5)
    print_graph(graph)
