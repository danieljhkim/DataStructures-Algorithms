package leetcode.src.com.leetcode;

import java.util.Stack;

public class StackSolutions {

    public static void main(String[] args) {
        System.out.println("");
    }

    // https://leetcode.com/problems/valid-parentheses/?envType=study-plan-v2&envId=top-interview-150
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(') {
                stack.push(')');
            } else if (c == '{') {
                stack.push('}');
            } else if (c == '[') {
                stack.push('[');
            } else if (stack.isEmpty() || stack.pop() != c) {
                return false;
            }
        }
        return stack.isEmpty();
    }
}
