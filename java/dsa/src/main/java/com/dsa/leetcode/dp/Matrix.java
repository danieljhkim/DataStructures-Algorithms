package com.dsa.leetcode.dp;

import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("all")
public class Matrix {

    int MAX_VAL = Integer.MAX_VALUE;

    // 931. Minimum Falling Path Sum
    public class P931 {

        int N;
        Map<String, Integer> memo = new HashMap<>();
        int[][] directions = { { 1, 0 }, { 1, 1 }, { 1, -1 } };

        public int minFallingPathSum(int[][] matrix) {
            this.N = matrix.length;
            int res = MAX_VAL;
            for (int i = 0; i < this.N; i++) {
                res = Math.min(res, dp(0, i, matrix));
            }
            return res;
        }

        int dp(int r, int c, int[][] matrix) {
            if (memo.containsKey(r + "," + c)) {
                return memo.get(r + "," + c);
            }
            int val = matrix[r][c];
            if (r == this.N - 1) {
                return val;
            }
            int res = MAX_VAL;
            for (int i = 0; i < 3; i++) {
                int nr = this.directions[i][0] + r;
                int nc = this.directions[i][1] + c;
                if (0 <= nr && nr < this.N && 0 <= nc && nc < this.N) {
                    int out = dp(nr, nc, matrix);
                    res = Math.min(out, res);
                }
            }
            memo.put(r + "," + c, val + res);
            return val + res;
        }
    }

    // 3459. Length of Longest V-Shaped Diagonal Segment
    public class P3459 {

        int R;
        int C;
        int[][] grid;
        static Integer[][][][][] memo;
        static final Map<Integer, int[]> dirs = new HashMap<>();
        static {
            dirs.put(-99, new int[] { 1, 1 });
            dirs.put(99, new int[] { -1, -1 });
            dirs.put(101, new int[] { 1, -1 });
            dirs.put(-101, new int[] { -1, 1 });
        }

        public int lenOfVDiagonal(int[][] grid1) {
            R = grid1.length;
            C = grid1[0].length;
            memo = new Integer[R][C][2][3][3];
            grid = grid1;
            int res = 0;
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    if (grid[r][c] == 1) {
                        for (int[] dir : dirs.values()) {
                            res = Math.max(res, dp(r, c, 0, dir));
                        }
                    }
                }
            }
            return res;
        }

        int dp(int r, int c, int turned, int[] dir) {
            if (memo[r][c][turned][dir[0] + 1][dir[1] + 1] != null) {
                return memo[r][c][turned][dir[0] + 1][dir[1] + 1];
            }
            int nxt = grid[r][c] == 2 ? 0 : 2;
            int nr = r + dir[0];
            int nc = c + dir[1];
            int res = 0;
            if (0 <= nr && nr < R && 0 <= nc && nc < C && nxt == grid[nr][nc]) {
                res = dp(nr, nc, turned, dir);
            }
            if (turned == 0) {
                int[] ndir = dirs.get(dir[0] * 100 + dir[1]);
                nr = r + ndir[0];
                nc = c + ndir[1];
                if (0 <= nr && nr < R && 0 <= nc && nc < C && nxt == grid[nr][nc]) {
                    res = Math.max(res, dp(nr, nc, 1, ndir));
                }
            }
            memo[r][c][turned][dir[0] + 1][dir[1] + 1] = res + 1;
            return res + 1;
        }
    }

}
