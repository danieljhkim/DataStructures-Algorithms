package com.dsa.algorithms.array;

import java.util.ArrayList;
import java.util.List;

public class BasicOperations {

    public void reverseArray(int[] array) {
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            left++;
            right--;
        }
    }

    public List<Integer> mergeSort(List<Integer> arr) {
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
