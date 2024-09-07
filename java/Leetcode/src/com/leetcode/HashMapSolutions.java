package com.leetcode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HashMapSolutions {

    public static void main(String[] args) {

        int[] map1 = new int[256];
        String st = "abc";
        map1[st.charAt(0)] = 5;
        System.out.println(map1[st.charAt(0)]);
        boolean ans = isIsomorphic("abc", "dga");
        System.out.println(ans);
    }

    // https://leetcode.com/problems/ransom-note/solutions/3744949/easy-solution-using-hashmap/?envType=study-plan-v2&envId=top-interview-150
    public static boolean canConstruct(String ransomNote, String magazine) {
        HashMap<Character, Integer> dictionary = new HashMap<>();

        for (int i = 0; i < magazine.length(); i++) {
            char c = magazine.charAt(i);
            if (!dictionary.containsKey(c)) {
                dictionary.put(c, 1);
            } else {
                dictionary.put(c, dictionary.get(c) + 1);
            }
        }

        for (int i = 0; i < ransomNote.length(); i++) {
            char c = ransomNote.charAt(i);
            if (!dictionary.containsKey(c) || dictionary.get(c) < 1) {
                return false;
            } else {
                dictionary.put(c, dictionary.get(c) - 1);
            }
        }
        return true;
    }

    // https://leetcode.com/problems/isomorphic-strings/?envType=study-plan-v2&envId=top-interview-150
    public static boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        int[] map1 = new int[256];
        int[] map2 = new int[256];

        for (int i = 0; i < s.length(); i++) {
            if (map1[s.charAt(i)] != map2[t.charAt(i)]) {
                return false;
            }
            map1[s.charAt(i)] = i + 1;
            map2[t.charAt(i)] = i + 1;
        }
        return true;
    }

    // https://leetcode.com/problems/two-sum/submissions/1152947535/?envType=study-plan-v2&envId=top-interview-150
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hmap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            hmap.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int comp = target - nums[i];
            if (hmap.containsKey(comp) && hmap.get(comp) != i) {
                int[] ans = new int[] {i, hmap.get(comp)};
                return ans;
            }
        }
        return new int[] {};
    }


    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>();
        if (digits.length() == 0) {
            return ans;
        }
        ans.add("");
        Map<Character, String> dMap = new HashMap<>();
        dMap.put('2', "abc");
        dMap.put('3', "def");
        dMap.put('4', "ghi");
        dMap.put('5', "jkl");
        dMap.put('6', "mno");
        dMap.put('7', "pqrs");
        dMap.put('8', "tuv");
        dMap.put('9', "wxyz");
        for (char c : digits.toCharArray()) {
            String s = dMap.get(c);
            List<String> comb = new ArrayList<>();
            for (int i = 0; i < ans.size(); i++) {
                String comStr = ans.get(i);
                for (char cc : s.toCharArray()) {
                    comb.add(comStr + cc);
                }
            }
            ans = comb;
        }
        return ans;
    }



}
