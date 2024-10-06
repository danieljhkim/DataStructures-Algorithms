package leetcode.src.com.dataStructures.stack;
import java.util.PriorityQueue;

public class MinStack {

    private static class StackNode {
        int val;
        StackNode next;

        public StackNode(int val) {
            this.val = val;
        }
    }

    StackNode top;
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();

    public MinStack() {
        
    }
    
    public void push(int val) {
        StackNode newStack = new StackNode(val);
        this.minHeap.add(val);
        if (this.top == null) {
            this.top = newStack;
        } else {
            StackNode temp = this.top;
            this.top = newStack;
            this.top.next = temp;
        }
    }
    
    public void pop() {
        if (this.top != null) {
            this.minHeap.remove(this.top.val);
            this.top = this.top.next;
        }
    }
    
    public int top() {
        if (this.top != null) {
            return this.top.val;
        }
        return -1;
    }
    
    public int getMin() {
        return this.minHeap.peek();
    }
}
