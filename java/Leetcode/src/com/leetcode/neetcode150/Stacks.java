package leetcode.src.com.leetcode.neetcode150;

import java.util.Stack;
import java.util.*;

public class Stacks {

    // 22. Generate Parentheses
    public List<String> generateParenthesis(int n) {
        Stack<String[]> stack = new Stack<>();
        List<String> ans = new ArrayList<>();
        String[] stuff = {"(", Integer.toString(n - 1), Integer.toString(n)};
        stack.add(stuff);
        while (!stack.isEmpty()) {
            String[] stuff1 = stack.pop();
            int open = Integer.parseInt(stuff1[1]);
            int close = Integer.parseInt(stuff1[2]);
            String ss = stuff1[0];
            if (open == 0 && close == 0) {
                ans.add(ss);
                continue;
            }
            if (open > 0) {
                stack.add(new String[] {ss + "(", Integer.toString(open - 1),
                        Integer.toString(close)});
            }
            if (open < close) {
                stack.add(new String[] {ss + ")", Integer.toString(open),
                        Integer.toString(close - 1)});
            }
        }
        return ans;
    }
}
