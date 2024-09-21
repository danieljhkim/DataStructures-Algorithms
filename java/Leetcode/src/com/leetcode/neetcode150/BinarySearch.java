package leetcode.src.com.leetcode.neetcode150;

import java.util.*;

public class BinarySearch {


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
}
