package leetcode.src.com.leetcode.neetcode150;

import java.util.*;

public class BinarySearch {

    // 704. Binary Search
    public int search(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            cur = nums[mid];
            if (cur < target) {
                low = mid + 1;
            } else if (cur > target) {
                high = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }


    // 74. Search a 2D matrix
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length;
        int col = matrix[0].length;

        int left = 0;
        int right = row * col - 1;
        while (left <= right) {
            int mid = (right + left) / 2;
            int value = matrix[mid / col][mid % col];
            if (value == target) {
                return true;
            } else {
                if (value > target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return false;
    }

    // 875. Koko Eating Bananas
    public int minEatingSpeed(int[] piles, int h) {
        int low = 1;
        int high = Arrays.stream(piles).max().orElse(0);
        while (low < high) {
            int k = low + (high - low) / 2;
            if (this._isFatty(k, h, piles)) {
                high = k;
            } else {
                low = k + 1;
            }
        }
        return low;
    }

    private boolean _isFatty(int k, int h, int[] piles) {
        int time = 0;
        for (int pile : piles) {
            time += (pile + k - 1) / k;
            if (time > h) return false; 
        }
        return true;
    }

    // 153. Find Minimum in Rotated Sorted Array
    public int findMin(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        if (high == 0 || (nums[low] < nums[high])) {
            return nums[0];
        }
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] > nums[high]) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return nums[low];
    }
}
