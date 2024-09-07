package Leetcode.src.com.leetcode.neetcode150;
import java.util.*;

public class ArraysHashing {
    
    // 217. Contains Duplicate
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> nset = new HashSet<>();
        for (int num : nums) {
            if (nset.contains(num)) {
                return true;
            }
            nset.add(num);
        }
        return false;
    }

    // 242. Valid Anagram
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        Map<Character, Integer> smap = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char ss = s.charAt(i);
            char tt = t.charAt(i);
            smap.put(ss, smap.getOrDefault(ss, 0) + 1);
            smap.put(tt, smap.getOrDefault(tt, 0) - 1);
        }
        for (int num : smap.values()) {
            if (num != 0) {
                return false;
            }
        }
        return true;
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> smap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int need = target - nums[i];
            if (smap.containsKey(need)) {
                return new int[] {i, smap.get(need)};
            }
            smap.put(nums[i], i);
        }
        return null;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs.length == 0) {
            return new ArrayList<List<String>>();
        }
        Map<String, List<String>> smap = new HashMap<>();
        int[] count = new int[26];
        for (String ss : strs) {
            Arrays.fill(count, 0);
            for (int i = 0; i < ss.length(); i++) {
                count[ss.charAt(i)-'a']++;
            }
            StringBuilder sb = new StringBuilder("");
            for (int i = 0; i < 26; i++) {
                sb.append('#');
                sb.append(count[i]);
            }
            String key = sb.toString();
            if (!smap.containsKey(key)) {
                smap.put(key, new ArrayList<>());
            }
            smap.get(key).add(ss);
        }
        return new ArrayList<>(smap.values());
    }


    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Set<Integer> set = new HashSet<>();
        int ans = 1;
        for (int num : nums) {
            set.add(num);
        }
        for (int num : set) {
            if (!set.contains(num-1)) {
                int len = 1;
                int next = num + 1;
                while (set.contains(next)) {
                    len += 1;
                    next += 1;
                }
                ans = Math.max(ans, len);
            }
        }
        return ans;
    }
}
