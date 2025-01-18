from collections import deque, defaultdict

def find_shortest_paths_dag(graph, start, target):
    dist = {start: 0}
    parents = defaultdict(list)
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                parents[nxt].append(node)
                queue.append(nxt)
            elif dist[nxt] == dist[node] + 1:
                parents[nxt].append(node)
    
    if target not in dist:
        return []
    
    paths = []
    def backtrack(u, path):
        if u == start:
            paths.append(path[::-1])
            return
        for p in parents[u]:
            backtrack(p, path + [p])
    
    backtrack(target, [target])
    return paths