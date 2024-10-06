package leetcode.src.com.leetcode.neetcode150;
import java.util.*;

public class ListNode {
    int val;
    ListNode next;

    ListNode() {}

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class LinkedList {
    
        // 21. Merge Two Sorted Lists
        public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
            ListNode ans = new ListNode();
            ListNode cur = ans;
            while (list1 != null && list2 != null) {
                if (list1.val > list2.val) {
                    ans.next = new ListNode(list2.val);
                    ans = ans.next;
                    list2 = list2.next;
                } else {
                    ans.next = new ListNode(list1.val);
                    ans = ans.next;
                    list1 = list1.next;
                }
            }
            ans.next = list1 == null ? list2 : list1;
            return cur.next;
        }
    
        // 19. Remove Nth Node From End of List
        public ListNode removeNthFromEnd(ListNode head, int n) {
            ListNode slow = head;
            ListNode fast = head;
            for (int i = 0; i < n; i++) {
                fast = fast.next;
            }
            if (fast.next == null) {
                return fast;
            }
            while (fast.next != null) {
                fast = fast.next;
                slow = slow.next;
            }
            slow.next = slow.next.next;
            return head;
        }

        // 206. Reverse Linked List
        public ListNode reverseList(ListNode head) {
            if (head == null || head.next == null) {
                return head;
            }
            ListNode prev = null;
            while (head != null) {
                ListNode temp = head.next;
                head.next = prev;
                prev = head;
                head = temp;
            }
            return prev;
        }

        // 143. Reorder List
        public void reorderList(ListNode head) {
            if (head.next == null) return;
            Stack<ListNode> stack = new Stack<>();
            ListNode fast = head;
            ListNode slow = head;
            while (fast != null && fast.next != null) {
                fast = fast.next.next;
                slow = slow.next;
            }

            ListNode mid = slow;
            slow = slow.next;
            mid.next = null; // Break the list into two halves

            while (slow != null) {
                stack.push(slow);
                slow = slow.next;
            }
            ListNode cur = head;
            while (!stack.isEmpty()) {
                ListNode tail = stack.pop();
                ListNode next = cur.next;
                cur.next = tail;
                tail.next = next;
                cur = next;
            }
        }

        // 287. Find the Duplicate Number
        public int findDuplicate(int[] nums) {
            // 3 4 1 4 2
            int slow = nums[0];
            int fast = nums[0];
            while (true) {
                slow = nums[slow];
                fast = nums[nums[fast]];
                if (slow == fast) break; 
            }
            slow = nums[0];
            while (slow != fast) {
                slow = nums[slow];
                fast = nums[fast];
            }
            return slow;
        }

}
