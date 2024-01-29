package com.leetcode;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
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
    
    //https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/
    public int countPairs(List<Integer> nums, int target) {
        int ans = 0;
        for (int i=0; i<nums.size()-1; i++) {
        	int ii = nums.get(i);
        	for (int j=i+1; j<nums.size(); j++) {
        		if ((ii + nums.get(j)) < target) {
        			ans++;
        		}
        	}
        }
        return ans;
    }
    
    //https://leetcode.com/problems/left-and-right-sum-differences/
    public int[] leftRightDifference(int[] nums) {
    	int len = nums.length;
    	int lsum = nums[0];
    	int rsum = nums[len-1];
    	int[] left = new int[len];
    	int[] right = new int[len];
    	int[] ans = new int[len];
    	left[0] = 0;
    	right[len-1] = 0;
    	for (int i=1; i<len; i++) {
    		left[i] = lsum;
    		lsum += nums[i];
    	}
    	for (int i=len-2; i>=0; i--) {
    		right[i] = rsum;
    		rsum += nums[i];
    	}
    	for (int i=0; i<len; i++) {
    		ans[i] = Math.abs(left[i] - right[i]);
    	}
    	return ans;
    }
    
    // https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
    public List<Integer> findDisappearedNumbers(int[] nums) {
        HashSet<Integer> hs = new HashSet<>();
        for(int val : nums)
          hs.add(val);
      
        List<Integer> al = new ArrayList<>();
        int n = nums.length;
        for(int i=1; i<=n; i++) {
            if(hs.contains(i) == false)
                al.add(i);
        }
        return al;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}