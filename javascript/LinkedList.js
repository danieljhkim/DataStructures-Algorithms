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

var findDuplicate = function(nums) {
  nums.sort();
  for(let i=0; i<nums.length-1; i++) {
    if(nums[i]===nums[i+1]) return nums[i];
  }
};

var wordPattern = function(pattern, s) {
  const ar = s.split(" ");
  if(pattern.length !== ar.length) return false;
  const keyMap = {};
  const valueMap = {};
  for(let i=0; i<ar.length; i++) {
    const key = pattern[i];
    const val = ar[i];
    if(key in keyMap || val in valueMap) {
      if(keyMap[key] !== val) return false;
      if(valueMap[val] !== key) return false;
    } else {
      keyMap[key] = val;
      valueMap[val] = key;
    }
  }
  return true;
};

var dayOfTheWeek = function(day, month, year) {
  const weekday = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];

  const dateObj = new Date(year, month, day);
  return weekday[dateObj.getDay()];
};

var isPrefixOfWord = function(sentence, searchWord) {
  let ar = sentence.split(" ");
  let len = searchWord.length;
  for(let i=0; i<ar.length; i++) {
    let word = ar[i].splice(0, len);
    if(word === searchWord) return i;
  }
  return -1;
};

var findDuplicate = function(nums) {
  let slow = nums[0];
  let fast = nums[nums[0]];

  while (slow !== fast) {
    slow = nums[slow];
    fast = nums[nums[fast]];
    console.log("slow", slow)
    console.log("fast", fast)
  }

  slow = 0;
  while(slow !== fast) {
    slow = nums[slow];
    fast = nums[fast];
    console.log("slowss", slow)
    console.log("fastss", fast)
  }
  return slow;
};