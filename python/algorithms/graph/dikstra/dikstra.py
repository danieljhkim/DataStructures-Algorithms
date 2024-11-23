import heapq
from typing import Dict
from math import inf


class Dikstra:

    ex_graph = {
        "A": {"B": 4, "C": 1},
        "B": {"C": 2, "D": 5},
        "C": {"D": 8, "E": 10},
        "D": {"E": 2},
        "E": {},
    }

    def __init__(self, graph=None):
        self.graph = graph if graph is not None else self.ex_graph

    def shortest_to_each_nodes(self, start_node: str):
        distances = {node: inf for node in self.graph}
        distances[start_node] = 0
        heap = [(0, start_node)]

        while heap:
            cur_dist, cur_node = heapq.heappop(heap)
            for neighbor, neighbor_dist in self.graph[cur_node].items():
                total_dist = neighbor_dist + cur_dist
                if total_dist < distances[neighbor]:
                    distances[neighbor] = total_dist
                    heapq.heappush(heap, (total_dist, neighbor))

        return distances

    def shortest_path_to_node(self, start_node: str, end_node: str):
        distances = {node: inf for node in self.graph}
        distances[start_node] = 0
        heap = [(0, start_node)]
        prev_nodes = {node: None for node in self.graph}

        while heap:
            cur_dist, cur_node = heapq.heappop(heap)

            if cur_node == end_node:  # reached the shortest path to destination
                break

            for neighbor, neighbor_dist in self.graph[cur_node].items():
                new_dist = neighbor_dist + cur_dist
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
                    prev_nodes[neighbor] = cur_node

        cur_pos = end_node
        paths = []
        while cur_pos:
            paths.append(cur_pos)
            cur_pos = prev_nodes[cur_pos]
        paths = paths[::-1]

        # when path not found
        if distances[end_node] == inf:
            return [], inf

        print(prev_nodes)
        return paths, distances[end_node]


if __name__ == "__main__":
    dikstra = Dikstra()
    path, distance = dikstra.shortest_path_to_node("A", "E")
    print(f"Path: {path}, Distance: {distance}")
