package com.dsa.leetcode.dp;

import java.util.Arrays;

@SuppressWarnings({"ReplaceStringBufferByString", "unused"})
public class TwoDimension {

    // 673. Number of Longest Increasing Subsequence
    class P673 {

        public int findNumberOfLIS(int[] nums) {
            int N = nums.length;
            int[] memoLen = new int[N];
            int[] memoCnt = new int[N];
            Arrays.fill(memoLen, 1);
            Arrays.fill(memoCnt, 1);
            int max = 1;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < i; j++) {
                    if (nums[i] > nums[j]) {
                        if (memoLen[i] < memoLen[j] + 1) {
                            memoLen[i] = memoLen[j] + 1;
                            memoCnt[i] = memoCnt[j];
                            max = Math.max(max, memoLen[i]);
                        } else if (memoLen[i] == memoLen[j] + 1) {
                            memoCnt[i] += memoCnt[j];
                        }
                    }
                }
            }
            int res = 0;
            for (int i = 0; i < N; i++) {
                if (memoLen[i] == max) {
                    res += memoCnt[i];
                }
            }
            return res;
        }
    }

    // 1143. Longest Common Subsequence
    class P1143 {

        int N1;
        int N2;
        String W1;
        String W2;
        Integer[][] memo;

        public int longestCommonSubsequence(String text1, String text2) {
            N1 = text1.length();
            N2 = text2.length();
            W1 = text1;
            W2 = text2;
            memo = new Integer[N1][N2];
            return dp(0, 0);
        }

        int dp(int idx1, int idx2) {
            if (idx1 == N1)
                return 0;
            if (idx2 == N2)
                return 0;
            if (memo[idx1][idx2] != null) {
              return memo[idx1][idx2];
            }
            char w1 = W1.charAt(idx1);
            char w2 = W2.charAt(idx2);
            int res = 0;
            if (w1 == w2) {
                res = dp(idx1 + 1, idx2 + 1) + 1;
            } else {
                res = Math.max(dp(idx1 + 1, idx2), dp(idx1, idx2 + 1));
            }
            memo[idx1][idx2] = res;
            return res;
        }
    }
}
