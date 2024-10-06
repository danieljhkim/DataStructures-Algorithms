package leetcode.src.com.leetcode.neetcode150;

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

}
