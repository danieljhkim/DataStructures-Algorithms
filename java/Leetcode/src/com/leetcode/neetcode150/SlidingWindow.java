package leetcode.src.com.leetcode.neetcode150;

public class SlidingWindow {

    public int maxProfit(int[] prices) {
        int profit = 0;
        int least = Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            least = Integer.min(least, prices[i]);
            int prof = prices[i] - least;
            if (prof > profit) {
                profit = prof;
            }
        }
        return profit;
    }
}
