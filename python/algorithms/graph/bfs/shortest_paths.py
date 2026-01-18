from collections import deque, defaultdict
import math


"""When BFS beats Dijkstra
- when you have exactly two distinct weights: 0 and any constant
"""


def min_cost_to_destination_bfs(grid: list):
    if not grid or not grid[0]:
        return 0

    R, C = len(grid), len(grid[0])
    dirs = ((0, 1), (1, 0), (-1, 0), (0, -1))
    dist = [[math.inf] * C for _ in range(R)]
    dist[0][0] = grid[0][0]

    dq = deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        cur = dist[r][c]

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                w = grid[nr][nc]
                nd = cur + w
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    if w == 0:  # IMPORTANT
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

    return dist[R - 1][C - 1]


"""Dial's Algorithm (Bucket BFS)
- O(E + V*W)
- all costs are integers in [0..W].
- Dial is only attractive when W is small (<20).
"""


def min_cost_to_destination_dial(grid: list):
    if not grid or not grid[0]:
        return 0
    R, C = len(grid), len(grid[0])
    V = R * C
    W = max(max(row) for row in grid)
    dirs = ((0, 1), (1, 0), (-1, 0), (0, -1))
    dist = [[math.inf] * C for _ in range(R)]
    dist[0][0] = grid[0][0]

    # Max possible shortest-path distance (simple upper bound)
    max_dist = grid[0][0] + W * (V - 1)

    buckets = [deque() for _ in range(max_dist + 1)]
    buckets[dist[0][0]].append((0, 0))

    # Process buckets in increasing distance order
    for d in range(dist[0][0], max_dist + 1):
        while buckets[d]:
            r, c = buckets[d].popleft()
            if d != dist[r][c]:  # stale
                continue
            if r == R - 1 and c == C - 1:
                return d

            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    nd = d + grid[nr][nc]
                    if nd < dist[nr][nc]:
                        dist[nr][nc] = nd
                        buckets[nd].append((nr, nc))

    return dist[R - 1][C - 1]


"""Bidirectional BFS
- explore from start and goal simultaneously
"""


def bidir_bfs_shortest_path_len(grid, start, goal):
    R, C = len(grid), len(grid[0])
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if start == goal:
        return 0

    # visited maps store distance from their respective source
    q1, q2 = deque([start]), deque([goal])
    d1, d2 = {start: 0}, {goal: 0}

    def expand(q, da, db):
        """Expand one BFS layer; return answer if meets other side."""
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if grid[nr][nc] == "#":
                    continue
                nxt = (nr, nc)
                if nxt in da:
                    continue
                da[nxt] = da[(r, c)] + 1
                if nxt in db:  # met the other search
                    return da[nxt] + db[nxt]
                q.append(nxt)
        return None

    while q1 and q2:
        # expand the smaller frontier for efficiency
        if len(q1) <= len(q2):
            ans = expand(q1, d1, d2)
        else:
            ans = expand(q2, d2, d1)
        if ans is not None:
            return ans

    return None  # no path


"""Multi-source BFS
- enqueue all sources initially
- distance to nearest x
"""


def multi_source_bfs_dist_to_nearest(grid, is_source):
    """
    Unweighted grid BFS.
    Returns dist[r][c] = min steps from (r,c) to nearest source cell.

    - grid: 2D indexable
    - is_source: callable (r,c)->bool marking sources
    """
    if not grid or not grid[0]:
        return []

    R, C = len(grid), len(grid[0])
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = [[math.inf] * C for _ in range(R)]
    q = deque()

    # enqueue all sources initially
    for r in range(R):
        for c in range(C):
            if is_source(r, c):
                dist[r][c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        d = dist[r][c]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                if dist[nr][nc] > d + 1:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc))

    return dist
