package com.dsa.leetcode.shotestPaths;

import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Queue;

public class Dijkstra {

    // 505. The Maze II
    public class P505 {

        public int shortestDistance(int[][] grid, int[] start, int[] dest) {
            int R = grid.length;
            int C = grid[0].length;
            int[][] dirs = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
            int[][] distance = new int[R][C];
            for (int[] row : distance) {
                Arrays.fill(row, Integer.MAX_VALUE);
            }
            Queue<int[]> queue = new PriorityQueue<>((a, b) -> a[2] - b[2]);
            queue.offer(new int[] { start[0], start[1], 0 });

            while (!queue.isEmpty()) {
                int[] cur = queue.poll();
                if (cur[0] == dest[0] && cur[1] == dest[1]) {
                    return cur[2];
                }
                for (int[] dir : dirs) {
                    int nr = cur[0];
                    int nc = cur[1];
                    int steps = 0;
                    while (nr + dir[0] >= 0 && nr + dir[0] < R
                            && nc + dir[1] >= 0 && nc + dir[1] < C
                            && grid[nr + dir[0]][nc + dir[1]] == 0) {
                        steps++;
                        nr += dir[0];
                        nc += dir[1];
                    }
                    int newDist = steps + cur[2];
                    if (distance[nr][nc] > newDist) {
                        distance[nr][nc] = newDist;
                        queue.offer(new int[] { nr, nc, newDist });
                    }
                }
            }
            return -1;
        }
    }
}
