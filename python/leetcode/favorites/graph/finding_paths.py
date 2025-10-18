"""
You are given a starting city and an unordered list of trips for a customer, you are tasked with creating an itinerary for that customer.
Trips are presented as pairs consisting of start and destination city names.
All cities must be used to produce the itinerary. If this is not possible return null.

Example 1:
Unordered list:
[["Amsterdam", "London"], ["Berlin", "Amsterdam"], ["Barcelona", "Berlin"], ["London", "Milan"]]

Starting:
"Barcelona"

Result:
["Barcelona", "Berlin", "Amsterdam", "London", "Milan"]
"""

from collections import defaultdict


def find_itinerary(flights: list, start: str):
    # visit each city only once
    adj = defaultdict(list)
    cities = set()
    for src, dst in flights:
        adj[src].append(dst)
        cities.add(src)
        cities.add(dst)

    N = len(cities)
    if len(adj) < N - 1:
        return None

    def dfs(cur, visited):
        if len(visited) == N:
            return [cur]
        for dst in adj[cur]:
            if dst not in visited:
                visited.add(dst)
                res = dfs(dst, visited)
                if res:
                    res.append(cur)
                    return res
                visited.remove(dst)
        return None

    res = dfs(start, {start})
    if res is None:
        return None
    res.reverse()
    return res


def find_itinerary2(flights: list, start: str):
    # could re-visit same city multiple times
    adj = defaultdict(list)
    outdegree = defaultdict(int)
    cities = set()
    for src, dst in flights:
        adj[src].append(dst)
        outdegree[src] += 1
        cities.add(src)
        cities.add(dst)

    def find_cycles(cur, start_city, visited):
        """
        for nodes with multiple out-going nodes and has cycle,
        we want to assign the max number of outdegree
        so that we can later put a constraint on how many times we can enter
        """
        if cur == start_city:
            return True
        for nei in adj[cur]:
            if nei not in visited:
                visited.add(nei)
                res = find_cycles(nei, start_city, visited)
                if res:
                    # how many times we can enter this node
                    outdegree[cur] = max(outdegree[nei], outdegree[start_city] - 1)
                    return res
                visited.remove(nei)
        return False

    for key, val in outdegree.items():
        if val > 1:
            for nei in adj[key]:
                find_cycles(nei, key, {nei})

    def dfs(cur, remaining: set):
        if not remaining:
            return [cur]
        if outdegree[cur] < 1:
            # revisiting it is redundant, so pass
            return None
        for nei in adj[cur]:
            res = None
            outdegree[nei] -= 1
            if nei in remaining:
                remaining.remove(nei)
                res = dfs(nei, remaining)
                remaining.add(nei)
            elif outdegree[nei] >= 0:
                res = dfs(nei, remaining)
            if res:
                return res + [cur]
            outdegree[nei] += 1
        return None

    res = dfs(start, cities)
    if res is None:
        return None
    res.reverse()
    return res


if __name__ == "__main__":
    a = [
        ["A", "B"],
        ["A", "E"],
        ["A", "G"],
        ["B", "C"],
        ["B", "Z"],
        ["E", "C"],
        ["G", "C"],
        ["C", "A"],
        ["Z", "Q"],
        ["Z", "A"],
        ["Q", "A"],
    ]
    res = find_itinerary2(a, "A")
    print(res)
