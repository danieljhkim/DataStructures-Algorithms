package com.dsa.leetcode.heap;

import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.PriorityQueue;

@SuppressWarnings({"ReplaceStringBufferByString", "unused"})
public class HeapSolutions {

    // 2462. Total Cost to Hire K Workers
    class P2462 {
        public long totalCost(int[] costs, int k, int candidates) {
            int N = costs.length;
            PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> {
                if (a[0] == b[0]) {
                    return a[1] - b[1];
                }
                return a[0] - b[0];
            });
            for (int i = 0; i < candidates; i++) {
                heap.offer(new int[] {costs[i], 0});
            }
            for (int i = Math.max(N - candidates, candidates); i < N; i++) {
                heap.offer(new int[] {costs[i], 1});
            }
            long ans = 0;
            int left = candidates;
            int right = N - 1 - candidates;

            while (k > 0) {
                int[] out = heap.poll();
                ans += out[0];
                if (left <= right) {
                    if (out[1] == 0) {
                        heap.offer(new int[]{costs[left], 0});
                        left += 1;
                    } else {
                        heap.offer(new int[]{costs[right], 1});
                        right -= 1;
                    }
                }
                k -= 1;
            }
            return ans;
        }
    }
}
