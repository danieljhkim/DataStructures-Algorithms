package com.leetcode;

import java.util.HashMap;

public class HashMapSolutions {
	
    public static void main(String[] args) {

        int[] map1 = new int[256];
        String st = "abc";
        map1[st.charAt(0)] = 5;
        System.out.println(map1[st.charAt(0)]);
        boolean ans = isIsomorphic("abc", "dga");
        System.out.println(ans);
    }
    
    //https://leetcode.com/problems/ransom-note/solutions/3744949/easy-solution-using-hashmap/?envType=study-plan-v2&envId=top-interview-150
    public static boolean canConstruct(String ransomNote, String magazine) {
        HashMap<Character, Integer> dictionary = new HashMap<>();
        
        for (int i=0; i < magazine.length(); i++) {
        	char c = magazine.charAt(i);
        	if (!dictionary.containsKey(c)) {
        		dictionary.put(c, 1);
        	} else {
        		dictionary.put(c, dictionary.get(c) + 1);
        	}
        }
        
        for (int i=0; i<ransomNote.length(); i++) {
        	char c = ransomNote.charAt(i);
        	if (!dictionary.containsKey(c) || dictionary.get(c) < 1) {
        		return false;
        	} else {
        		dictionary.put(c, dictionary.get(c) - 1);
        	}
        }
        return true;
    }
    
    //https://leetcode.com/problems/isomorphic-strings/?envType=study-plan-v2&envId=top-interview-150
    public static boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
        	return false;
        }
        
        int[] map1 = new int[256];
        int[] map2 = new int[256];
        
        for (int i=0; i<s.length(); i++) {
        	if (map1[s.charAt(i)] != map2[t.charAt(i)]) {
        		return false;
        	}
        	map1[s.charAt(i)] = i + 1;
        	map2[t.charAt(i)] = i + 1;
        }
        return true;
        
    }
    
    
    
    
    
    
    
    
    
    
    
}
