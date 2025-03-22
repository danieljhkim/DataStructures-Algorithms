package com.dsa.leetcode.shotestPaths;

import java.util.ArrayDeque;
import java.util.Deque;

@SuppressWarnings({"ReplaceStringBufferByString", "unused"})
public class Bfs {

    // 1197. Minimum Knight Moves
    public class P1197 {

        public int minKnightMoves(int x, int y) {
            int[][] moves = {
                {2, 1}, {1, 2}, {-2, 1}, {-1, 2}, {-2, -1}, {-1, -2}, {2, -1}, {1, -2}
            };
            boolean[][] visited = new boolean[607][607];
            Deque<int[]> queue = new ArrayDeque<>();
            queue.addLast(new int[] {0, 0, 0});

            while (!queue.isEmpty()) {
                int[] cur = queue.removeFirst();
                if (cur[1] == x && cur[2] == y) {
                    return cur[0];
                }
                for (int[] move : moves) {
                    int nr = cur[1] + move[0];
                    int nc = cur[2] + move[1];
                    if (!visited[nr + 302][nc + 302]) {
                        visited[nr + 302][nc + 302] = true;
                        queue.addLast(new int[] {cur[0] + 1, nr, nc});
                    }
                }
            }
            return -1;
        }
    }
}
