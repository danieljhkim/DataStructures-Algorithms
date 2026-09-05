"""
I found this question floating around, and while solving it unlocked a new concept for me.

Question:

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
from collections import Counter, defaultdict
from typing import List, Optional


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


def find_itinerary_wrong(flights: list, start: str):
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


"""
The three fixes
---------------
1. `dfs(start, cities)` left `start` inside `remaining`, so `if not remaining`
   could never fire unless a trip happened to lead back to the start. Gone
   entirely -- see 3.
 
2. `outdegree[nei] -= 1` charged the wrong city. Walking cur -> nei spends a
   trip out of *cur*. As written you arrived at a city having just docked its
   own budget and then failed your own `outdegree[cur] < 1` guard, so every
   ordinary chain link rejected itself. The fix is to consume the specific
   trip, not to decrement a city-wide counter.
 
3. The stopping condition was "every city visited", but the task is "every trip
   used" -- with [A->B, B->A, A->B] that halts at ['A','B'] having used one trip
   of three. Now it counts trips.
 
Why the counter had to go rather than be corrected: a per-city number cannot
say *which* of several parallel trips was spent. Patch it into consistency and
it still produces routes like ['A','B','C','B','C'] that reuse a trip. The
remaining structure below is `src -> Counter(dst -> trips left)`, which spends
one specific trip at a time and restores it on backtrack.
 
Counting destinations rather than listing them also prunes: k identical trips
A->B are one branch tried once, not k! orderings.
"""


def find_itinerary_fixed(flights: List[List[str]], start: str) -> Optional[List[str]]:
    total = len(flights)
    if total == 0:
        return [start]

    remaining = defaultdict(Counter)
    for src, dst in flights:
        remaining[src][dst] += 1

    route = [start]

    def dfs(cur: str, used: int) -> bool:
        if used == total:
            return True
        outgoing = remaining.get(cur)
        if not outgoing:
            return False
        for dst, left in list(outgoing.items()):
            if left == 0:
                continue
            outgoing[dst] = left - 1  # spend this trip
            route.append(dst)
            if dfs(dst, used + 1):
                return True
            route.pop()
            outgoing[dst] = left  # put it back

        return False

    return route if dfs(start, 0) else None


if __name__ == "__main__":
    a = [
        ["Amsterdam", "London"],
        ["Berlin", "Amsterdam"],
        ["Barcelona", "Berlin"],
        ["London", "Milan"],
    ]
    res = find_itinerary(a, "Barcelona")
    print(res)
