package leetcode.src.com.leetcode;

import java.util.*;

class Solutions {

	public static void main(String[] args) {

	}

	String increaseStr(String word) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < word.length(); i++) {
			char nextChar = 'a';
			if (word.charAt(i) != 'z') {
				nextChar = (char) (word.charAt(i) + 1);
			}
			sb.append(nextChar);
		}
		return sb.toString();
	}
	// 3304. Find the K-th Character in String Game I
	public char kthCharacter(int k) {
		StringBuilder word = new StringBuilder("a");
		while (word.length() < k) {
			word.append(increaseStr(word.toString()));
			System.out.println(word.toString());
		}
		return word.charAt(k - 1);
	}

	// 3307. Find the K-th Character in String Game II
	public char kthCharacter(long k, int[] operations) {
		StringBuilder word = new StringBuilder("a");
		for (int i : operations) {
			if (i == 0) {
				word.append(word.toString());
			} else {
				word.append(increaseStr(word.toString()));
			}
		}
		return word.charAt(k - 1);
    }

	boolean isVowelsFull(Map<Character, Integer> vowelMap) {
		for (Integer value : vowelMap.values()) {
			if (value < 1) return false;
		}
		return true;
	}

	// 3305. Count of Substrings Containing Every Vowel and K Consonants I
	public int countOfSubstrings(String word, int k) {
		int n = word.length();
		int ans = 0;
		int left = 0;
		while (left < n) {
			Map<Character, Integer> vowelMap = new HashMap<>();
			vowelMap.put('a', 0);
			vowelMap.put('e', 0);
			vowelMap.put('i', 0);
			vowelMap.put('o', 0);
			vowelMap.put('u', 0);
			int cons = 0;
			int right = left;
			while (right < n) {
				char curChar = word.charAt(right);
				if (vowelMap.containsKey(curChar)) {
					vowelMap.put(curChar, vowelMap.get(curChar) + 1);
				} else {
					cons++;
					if (cons > k) {
						break;
					}
				}
				if (isVowelsFull(vowelMap) && cons == k) {
					ans++;
				}
				right++;
			}
			left++;
		}
		return ans;
    }


}
