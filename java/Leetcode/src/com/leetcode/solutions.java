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
    			System.out.println(c);
    			str.append(c);
    		}
    	}
    	String notRev = str.toString().toLowerCase();
    	String rev = str.reverse().toString().toLowerCase();
    	System.out.println(rev);
    	System.out.println(notRev);
    	if (rev.equals(notRev)) {
    		return true;
    	}
    	return false;
    }
}