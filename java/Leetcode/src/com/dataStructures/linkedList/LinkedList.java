package leetcode.src.com.dataStructures.linkedList;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LinkedList {
	
	public int getDecimalValue(ListNode head) {
		List<Integer> li = new ArrayList<>();
		while (head != null) {
			li.add(head.val);
			head = head.next;
		}
		Collections.reverse(li);
		int ans = 0;
		for (int i = 0; i < li.size(); i++) {
			ans += li.get(i) * Math.pow(2, i);
		}
		return ans;
	}

	public ListNode reverse(ListNode head, ListNode node) {
		ListNode prev = null;
		ListNode next = null;
		ListNode curr = head;
		while (node != null) {
			next = curr.next;
			curr.next = prev;
			prev = curr;
			curr = next;
		}
		return prev;
	}

	public void deleteNode(ListNode node) {
		node.val = node.next.val;
		node.next = node.next.next;
	}
}


