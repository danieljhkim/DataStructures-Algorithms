package com.dsa.leetcode.dp;

import java.util.HashMap;
import java.util.Map;

public class Matrix {

    int MAX_VAL = Integer.MAX_VALUE;
    
    // 931. Minimum Falling Path Sum
    public class P931 {
        
        int N;
        Map<String, Integer> memo = new HashMap<>();
        int[][] directions = { {1, 0}, {1, 1}, {1, -1} };

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
}
