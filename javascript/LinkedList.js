/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */

 var deleteDuplicates = function(head) {
  let node = head;
  while(node) {
    if(!node.next) break;
    if(node.val === node.next.val) {
      node.next = node.next.next;
    } else {
      node = node.next;
    }
  }
  return head;
};

//https://leetcode.com/problems/remove-linked-list-elements/
var removeElements = function(head, val) {
  let node = head;
  while(node && node.next !== null) {
    if(!node.next.next && node.next.val === val) {
      node.next = null;
      break;
    }
    if(node.val === val) {
      node.val = node.next.val;
      node.next = node.next.next;
    } else {
      node = node.next;
    }
  }
  return head.val===val?head.next:head;
};
