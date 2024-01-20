package com.leetcode;

import java.util.HashMap;

public class HashMapSolutions {
	
    public static void main(String[] args) {
    	boolean ans = canConstruct("aa", "aba");
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
}
