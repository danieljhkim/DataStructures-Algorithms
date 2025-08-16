package com.dsa.leetcode.practice;

import com.dsa.leetcode.practice.Solutions1.TreeNode;

import java.util.*;

@SuppressWarnings({"ReplaceStringBufferByString", "unused"})
public class Solutions1 {

    int MAX_VAL = Integer.MAX_VALUE;

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {}

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

    // 1657. Determine if Two Strings Are Close
    class P1657 {

        public boolean closeStrings(String word1, String word2) {
            int N1 = word1.length();
            int N2 = word2.length();
            if (N1 != N2) return false;
            Set<Character> set1 = new HashSet<>();
            Set<Character> set2 = new HashSet<>();
            for (char c : word1.toCharArray()) {
                set1.add(c);
            }
            for (char c : word2.toCharArray()) {
                set2.add(c);
            }
            if (!set1.equals(set2)) {
                return false;
            }
            int[] count1 = new int[26];
            int[] count2 = new int[26];

            for (int i = 0; i < N1; i++) {
                int idx1 = word1.codePointAt(i) - 'a';
                int idx2 = word2.codePointAt(i) - 'a';
                count1[idx1] += 1;
                count2[idx2] += 1;
            }
            Arrays.sort(count1);
            Arrays.sort(count2);
            for (int i = 0; i < 26; i++) {
                int cnt1 = count1[i];
                int cnt2 = count2[i];
                if (cnt1 != cnt2) {
                    return false;
                }
            }
            return true;
        }
    }

    class P2226 {

        public int maximumCandies(int[] candies, long k) {
            int low = 1;
            int high = (int) Math.pow(10, 7);

            while (low <= high) {
                int mid = (high + low) / 2;
                if (isEnough(mid, k, candies)) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            return high;
        }

        boolean isEnough(int amount, long k, int[] candies) {
            long num = 0;
            for (int c : candies) {
                if (c >= amount) {
                    num += c / amount;
                    if (num >= k) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    // 2594. Minimum Time to Repair Cars
    class P2594 {

        public long repairCars(int[] ranks, int cars) {
            // time = r * n^2
            // n^2 = time / r
            long low = 1;
            long high = (long) 100 * cars * cars;
            while (low <= high) {
                long mid = (low + high) / 2;
                if (enough(mid, cars, ranks)) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }
            return low;
        }

        boolean enough(long time, int cars, int[] ranks) {
            long cnt = 0;
            for (int r : ranks) {
                cnt += (long) Math.floor(Math.sqrt((time / (long) r)));
            }
            return cnt >= cars;
        }
    }

    // 266. Palindrome Permutation
    class P266 {
        public boolean canPermutePalindrome(String s) {
            int[] counts = new int[26];
            int N = s.length();
            for (int i = 0; i < s.length(); i++) {
                counts[s.codePointAt(i) - 'a']++;
            }
            boolean isEven = N % 2 == 0;
            for (int c : counts) {
                if (c % 2 == 1) {
                    if (!isEven) {
                        isEven = true;
                    } else {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    // 2206. Divide Array Into Equal Pairs
    class P2206 {

        public boolean divideArray(int[] nums) {
            Set<Integer> nset = new HashSet<>();
            for (int i = 0; i < nums.length; i++) {
                if (nset.contains(nums[i])) {
                    nset.remove(nums[i]);
                } else {
                    nset.add(nums[i]);
                }
            }
            return nset.size() == 0;
        }
    }

    // 543. Diameter of Binary Tree
    class P543 {
        public int diameterOfBinaryTree(TreeNode root) {
            if (root == null) {
                return 0;
            }
            int[] ans = dfs(root);
            return Math.max(ans[0], ans[1]) - 1;
        }

        int[] dfs(TreeNode node) {
            if (node == null) {
                return new int[] {0, 0};
            }
            int[] left = dfs(node.left);
            int[] right = dfs(node.right);
            int maxLen = Math.max(left[0], right[0]);
            int maxInd = Math.max(left[1], right[1]);
            maxInd = Math.max(maxInd, left[0] + right[0] + 1);
            return new int[] {maxLen + 1, maxInd};
        }
    }

    // 1249. Minimum Remove to Make Valid Parentheses
    class P1249 {
        public String minRemoveToMakeValid(String s) {
            StringBuilder sb = new StringBuilder();
            int left = 0;
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == '(') {
                    left++;
                } else if (s.charAt(i) == ')') {
                    if (left > 0) {
                        left--;
                    } else {
                        continue;
                    }
                }
                sb.append(s.charAt(i));
            }

            int idx = sb.length() - 1;
            StringBuilder sb2 = new StringBuilder();
            while (idx >= 0) {
                if (sb.charAt(idx) == '(') {
                    if (left > 0) {
                        left--;
                        idx--;
                        continue;
                    }
                }
                sb2.append(sb.charAt(idx));
                idx--;
            }
            return sb2.reverse().toString();
        }
    }

    // 1004. Max Consecutive Ones III
    class P1004 {
        public int longestOnes(int[] nums, int k) {
            int left = 0, right = 0, zeros = 0, N = nums.length;
            while (right < N) {
                if (nums[right] == 0) {
                    zeros++;
                }
                if (zeros > k) {
                    if (nums[left] == 0) {
                        zeros--;
                    }
                    left++;
                }
                right++;
            }
            return right - left;
        }
    }

    // 198. House Robber
    class P198 {

        int N;
        int[] nums;
        Integer[][] memo;

        public int rob(int[] nums) {
            this.N = nums.length;
            this.nums = nums;
            this.memo = new Integer[2][N];
            return dp(0, 0);
        }

        public int dp(int idx, int robbed) {
            if (idx >= N) {
                return 0;
            }
            if (memo[robbed][idx] != null) {
                return memo[robbed][idx];
            }
            int res = 0;
            if (robbed == 1) {
                res = dp(idx + 1, 0);
            } else {
                res = Math.max(dp(idx + 1, 0), dp(idx + 1, 1) + nums[idx]);
            }
            memo[robbed][idx] = res;
            return res;
        }
    }

    // 2942. Find Words Containing Character
    class P2942 {

        public List<Integer> findWordsContaining(String[] words, char x) {
            List<Integer> res = new ArrayList<>();
            String target = x + "";
            for (int i = 0; i < words.length; i++) {
                if (words[i].contains(target)) {
                    res.add(i);
                }
            }
            return res;
        }
    }

    // 3619. Count Islands With Total Value Divisible by K
    class P3619 {

        int R;
        int C;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int[][] grid;

        public int countIslands(int[][] grid, int k) {
            R = grid.length;
            C = grid[0].length;
            this.grid = grid;
            int res = 0;

            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    if (grid[r][c] > 0) {
                        int val = dfs(r, c);
                        if (val % k == 0) {
                            res += 1;
                        }
                    }
                }
            }
            return res;
        }

        int dfs(int r, int c) {
            int val = grid[r][c];
            grid[r][c] = 0;
            for (int[] d : dirs) {
                int nr = d[0] + r;
                int nc = d[1] + c;
                if (0 <= nr && nr < R && 0 <= nc && nc < C && grid[nr][nc] > 0) {
                    val += dfs(nr, nc);
                }
            }
            return val;
        }
    }

    // 3627. Maximum Median Sum of Subsequences of Size 3
    class P3627 {
        public long maximumMedianSum(int[] nums) {
            Arrays.sort(nums);
            int cnt = nums.length / 3, idx = nums.length - 2;
            long res = 0;
            while (cnt > 0) {
                res += nums[idx];
                idx -= 2;
                cnt--;
            }
            return res;
        }
    }


}
