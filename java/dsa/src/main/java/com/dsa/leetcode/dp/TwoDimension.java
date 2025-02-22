package com.dsa.leetcode.dp;

import java.util.Arrays;

public class TwoDimension {

    // 673. Number of Longest Increasing Subsequence
    public class P673 {

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
}
