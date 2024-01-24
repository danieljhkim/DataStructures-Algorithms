package com.leetcode;

import java.util.ArrayList;
import java.util.List;

class Solution {
	
    public static void main(String[] args) {
        isPalindrome("race a car");
    }

	public int numberOfBeams(String[] bank) {
		int sum = 0;
		List<Integer> ar = new ArrayList<Integer>();
		if(bank.length == 1) return 0;
		for(String row : bank) {
			int beams = 0;
			for(char c : row.toCharArray()) {
				if(c == '1') {
					beams++;
				}
			}
			if(beams > 0) {
				ar.add(beams);
			}
		}
		int prev = 0;
		for(int j=0; j<ar.size()-1; j++) {
			sum += ar.get(j) * ar.get(j+1);
		}
		return sum;
	}
	
	
    public static boolean isPalindrome(String s) {
    	StringBuilder str = new StringBuilder();
    	for (char c : s.toCharArray()) {
    		if (Character.isLetterOrDigit(c)) {
    			str.append(c);
    		}
    	}
    	String notRev = str.toString().toLowerCase();
    	String rev = str.reverse().toString().toLowerCase();
    	if (rev.equals(notRev)) {
    		return true;
    	}
    	return false;
    }
    
    public List<Integer> findWordsContaining(String[] words, char x) {
    	List<Integer> ans = new ArrayList<>();
        for (int i=0; i<words.length; i++) {
        	for (char c : words[i].toCharArray()) {
        		if (c == x) {
        			ans.add(i);
        			break;
        		}
        	}
        }
        return ans;
    }
    
    public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int ans = 0;
        for (int h : hours) {
        	if (h >= target) {
        		ans++;
        	}
        }
        return ans;
    }
    
    //https://leetcode.com/problems/find-the-maximum-achievable-number/
    public int theMaximumAchievableX(int num, int t) {
        return t*2 + num;
    }
    
    //https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/
    public int differenceOfSums(int n, int m) {
    	int num1 = 0;
    	int num2 = 0;
        for (int i=1; i<=n; i++) {
        	if (i%m != 0) {
        		num1 += i;
        	} else {
        		num2 += i;
        	}
        }
        return num1-num2;
    }
    
    
}