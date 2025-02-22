package com.dsa.leetcode.practice;

import java.util.Stack;

public class Solutions1 {

    int MAX_VAL = Integer.MAX_VALUE;

    public static void main(String[] args) {
        System.out.println("Hello World!");
    }

    // 1861. Rotating the Box
    public class P1861 {

        public char[][] rotateTheBox(char[][] boxGrid) {
            int R = boxGrid.length;
            int C = boxGrid[0].length;
            char[][] grid = new char[C][R];

            for (int r = 0; r < R; r++) {
                int space = C - 1;
                for (int c = C - 1; c >= 0; c--) {
                    char val = boxGrid[r][c];
                    if (val == '#') {
                        grid[c][r] = '.';
                        grid[space][r] = '#';
                        space -= 1;
                    } else if (val == '*') {
                        grid[c][r] = '*';
                        space = c - 1;
                    } else {
                        grid[c][r] = '.';
                    }
                }
            }

            for (char[] row : grid) {
                reverse_row(row);
            }
            return grid;
        }

        void reverse_row(char[] row) {
            int left = 0;
            int right = row.length - 1;
            while (left < right) {
                char tmp = row[left];
                row[left] = row[right];
                row[right] = tmp;
                left++;
                right--;
            }
        }
    }

    // 487. Max Consecutive Ones II
    public class P487 {

        public int findMaxConsecutiveOnes(int[] nums) {
            int max_len = 0;
            int cur = 0;
            Stack<Integer> stack = new Stack<>();

            for (int n : nums) {
                if (n == 0) {
                    int prev = 0;
                    if (!stack.isEmpty()) {
                        prev = stack.pop() + 1;
                    }
                    stack.add(cur);
                    max_len = Math.max(max_len, cur + prev);
                    cur = 0;
                } else {
                    cur++;
                }
            }
            int prev = 0;
            if (!stack.isEmpty()) {
                prev = stack.pop() + 1;
            }
            max_len = Math.max(max_len, cur + prev);
            return max_len;
        }
    }

}
