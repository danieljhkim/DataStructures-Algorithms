package com.dsa.utils;

public class Bisect {

    public static int bisectLeft(int[] arr, int target, int low, int high) {
        while (low <= high) {
            int mid = (high + low) / 2;
            if (arr[mid] > target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public static int bisectLeft(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = (high + low) / 2;
            if (arr[mid] > target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    /**
     * Returns right-most insertion point
     */
    public static int bisectRight(int[] arr, int target, int low, int high) {
        while (low <= high) {
            int mid = (high + low) / 2;
            if (arr[mid] < target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
    public static int bisectRight(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = (high + low) / 2;
            if (arr[mid] < target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
}
