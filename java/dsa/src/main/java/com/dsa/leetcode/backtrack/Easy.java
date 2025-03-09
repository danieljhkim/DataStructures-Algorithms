package com.dsa.leetcode.backtrack;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Easy {

    public static void main(String[] args) {
        System.out.println("Hello World!");
    }

    // 1980. Find Unique Binary String
    public class P1980 {

        Set<String> nset;
        String ans;

        public String findDifferentBinaryString(String[] nums) {
            int n = nums.length;
            this.nset = new HashSet<>();
            Collections.addAll(nset, nums);
            backtrack("", n);
            return this.ans;
        }

        void backtrack(String value, int n) {
            if (this.ans != null) return;
            if (value.length() == n) {
                if (!nset.contains(value)) {
                    this.ans = value;
                }
                return;
            }
            backtrack(value + "0", n);
            backtrack(value + "1", n);
        }
    }

    // 1415
    public class P1415 {

        String options = "abc";
        List<String> ans = new ArrayList<>();

        public String getHappyString(int n, int k) {
            backtrack(new ArrayList<>(), n, k);
            if (ans.size() < k) {
                return "";
            }
            return ans.get(k - 1);
        }

        void backtrack(List<String> arr, int n, int k) {
            if (arr.size() == n) {
                ans.add(String.join("", arr));
                return;
            }
            if (ans.size() >= k) {
                return;
            }
            for (int i = 0; i < 3; i++) {
                String w = String.valueOf(options.charAt(i));
                if (arr.isEmpty() || (!arr.isEmpty() && !arr.get(arr.size() - 1).equals(w))) {
                    arr.add(w);
                    backtrack(arr, n, k);
                    arr.remove(arr.size() - 1);
                }
            }
        }
    }
}
