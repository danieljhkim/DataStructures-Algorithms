package com.dsa.leetcode.practice;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import com.dsa.leetcode.practice.Solutions1.TreeNode;

@SuppressWarnings({ "ReplaceStringBufferByString", "unused" })
public class Solutions1 {

    int MAX_VAL = Integer.MAX_VALUE;

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
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

    // 2023. Number of Pairs of Strings With Concatenation Equal to Target
    public class P2023 {

        public int numOfPairs(String[] nums, String target) {
            StringBuilder sb = new StringBuilder(target);
            Map<String, Integer> table = new HashMap<>();
            Set<Integer> nset = new HashSet<>();
            int cnt = 0;
            int N = target.length();

            for (String n : nums) {
                int len = n.length();
                if (len >= N - 1) {
                    continue;
                }
                table.put(n, table.getOrDefault(n, 0) + 1);
                nset.add(len);
            }
            for (int len : nset) {
                String left = sb.substring(0, len);
                String right = sb.substring(len);
                if (table.containsKey(left) && table.containsKey(right)) {
                    if (left.equals(right)) {
                        cnt += table.get(left) * (table.get(left) - 1);
                    } else {
                        cnt += table.get(left) * table.get(right);
                    }
                }
            }
            return cnt;
        }
    }

    // 2187. Minimum Time to Complete Trips
    class P2187 {

        int target;

        public long minimumTime(int[] time, int totalTrips) {
            long lo = 0;
            long hi = (long) time[0] * totalTrips;
            target = totalTrips;
            while (lo < hi) {
                long mid = (hi - lo) / 2 + lo;
                if (totalTrips(time, mid)) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            return hi;
        }

        boolean totalTrips(int[] time, long duration) {
            long trips = 0;
            for (int n : time) {
                trips += duration / n;
                if (trips >= target) {
                    return true;
                }
            }
            return false;
        }
    }

    // 2570. Merge Two 2D Arrays by Summing Values
    class P2570 {

        // static Random random = new Random();
        Map<Integer, Integer> map = new HashMap<>();
        int N;

        public int[][] mergeArrays(int[][] nums1, int[][] nums2) {
            for (int[] item : nums1) {
                map.put(item[0], item[1]);
            }
            for (int[] item : nums2) {
                map.put(item[0], map.getOrDefault(item[0], 0) + item[1]);
            }
            N = map.size();
            int[][] ans = new int[N][2];
            List<Integer> arr = new ArrayList<>();
            arr.addAll(map.keySet());
            arr = mergeSort(arr);
            for (int i = 0; i < N; i++) {
                int idx = arr.get(i);
                ans[i][0] = idx;
                ans[i][1] = map.get(idx);
            }
            return ans;
        }

        List<Integer> mergeSort(List<Integer> arr) {
            int len = arr.size();
            if (len <= 1) {
                return arr;
            }
            List<Integer> left = mergeSort(new ArrayList<>(arr.subList(0, len / 2)));
            List<Integer> right = mergeSort(new ArrayList<>(arr.subList(len / 2, len)));
            int l = 0, r = 0, k = 0;
            while (l < left.size() && r < right.size()) {
                if (left.get(l) < right.get(r)) {
                    arr.set(k, left.get(l));
                    l++;
                } else {
                    arr.set(k, right.get(r));
                    r++;
                }
                k++;
            }
            while (l < left.size()) {
                arr.set(k, left.get(l));
                l++;
                k++;
            }
            while (r < right.size()) {
                arr.set(k, right.get(r));
                r++;
                k++;
            }
            return arr;
        }
    }

}
