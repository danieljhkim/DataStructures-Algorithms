package leetcode.src.com.leetcode.neetcode150;

public class TwoPointers {

    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            char l = s.charAt(left);
            char r = s.charAt(right);
            if (!Character.isLetterOrDigit(l)) {
                left++;
                continue;
            }
            if (!Character.isLetterOrDigit(r)) {
                right--;
                continue;
            }
            if (l != r) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    // 167. Two Sum II - Input Array Is Sorted
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int total = numbers[left] + numbers[right];
            if (total < target) {
                left++;
            } else if (total > target) {
                right--;
            } else {
                return new int[] {left + 1, right + 1};
            }
        }
        return null;
    }
}
