package com.dsa.leetcode.shotestPaths;

import java.util.*;

@SuppressWarnings({"ReplaceStringBufferByString", "unused"})
public class Dijkstra {

    // 505. The Maze II
    public class P505 {

        public int shortestDistance(int[][] grid, int[] start, int[] dest) {
            int R = grid.length;
            int C = grid[0].length;
            int[][] dirs = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
            int[][] distance = new int[R][C];
            for (int[] row : distance) {
                Arrays.fill(row, Integer.MAX_VALUE);
            }
            Queue<int[]> queue = new PriorityQueue<>((a, b) -> a[2] - b[2]);
            queue.offer(new int[] {start[0], start[1], 0});

            while (!queue.isEmpty()) {
                int[] cur = queue.poll();
                if (cur[0] == dest[0] && cur[1] == dest[1]) {
                    return cur[2];
                }
                for (int[] dir : dirs) {
                    int nr = cur[0];
                    int nc = cur[1];
                    int steps = 0;
                    while (nr + dir[0] >= 0
                            && nr + dir[0] < R
                            && nc + dir[1] >= 0
                            && nc + dir[1] < C
                            && grid[nr + dir[0]][nc + dir[1]] == 0) {
                        steps++;
                        nr += dir[0];
                        nc += dir[1];
                    }
                    int newDist = steps + cur[2];
                    if (distance[nr][nc] > newDist) {
                        distance[nr][nc] = newDist;
                        queue.offer(new int[] {nr, nc, newDist});
                    }
                }
            }
            return -1;
        }
    }

    // 2737. Find the Closest Marked Node
    class P2737 {

        public int minimumDistance(int n, List<List<Integer>> edges, int s, int[] marked) {
            Map<Integer, List<int[]>> adj = new HashMap<>();
            for (List<Integer> edge : edges) {
                int u = edge.get(0);
                int v = edge.get(1);
                int w = edge.get(2);
                List<int[]> entry = adj.getOrDefault(u, new ArrayList<>());
                entry.add(new int[] {w, v});
                adj.put(u, entry);
            }

            Set<Integer> nset = new HashSet<>();
            for (int d : marked) {
                nset.add(d);
            }
            int[] distances = new int[n];
            Arrays.fill(distances, Integer.MAX_VALUE);
            distances[s] = 0;
            PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            pq.offer(new int[] {0, s});
            while (!pq.isEmpty()) {
                int[] cur = pq.poll();
                if (nset.contains(cur[1])) {
                    return cur[0];
                }
                List<int[]> nxt = adj.get(cur[1]);
                if (nxt != null && !nxt.isEmpty()) {
                    for (int[] nei : nxt) {
                        int ndist = nei[0] + cur[0];
                        if (distances[nei[1]] > ndist) {
                            distances[nei[1]] = ndist;
                            pq.offer(new int[] {ndist, nei[1]});
                        }
                    }
                }
            }
            return -1;
        }
    }
}
