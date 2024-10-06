package leetcode.src.com.leetcode.neetcode150;

import java.util.*;

public class Backtracking {

    // 78. Subsets
    public List<List<Integer>> subsets(int[] nums) {
        // iterative approach

        List<List<Integer>> ans = new ArrayList<>();
        Stack<Map.Entry<Integer, List<Integer>>> stack = new Stack<>();

        for (int size = 0; size <= nums.length; size++) {
            stack.push(new AbstractMap.SimpleEntry<>(0, new ArrayList<>()));
            while (!stack.isEmpty()) {
                Map.Entry<Integer, List<Integer>> curr = stack.pop();
                int idx = curr.getKey();
                List<Integer> curArr = curr.getValue();
                if (curArr.size() == size) {
                    ans.add(new ArrayList<>(curArr));
                    continue;
                }
                for (int i = idx; i < nums.length; i++) {
                    List<Integer> newArr = new ArrayList<>(curArr);
                    newArr.add(nums[i]);
                    stack.push(new AbstractMap.SimpleEntry<>(i + 1, newArr));
                }
            }
        }
        return ans;
    }

    public void subsetBacktrack(List<Integer> arr, int[] nums, int idx, int size,
            List<List<Integer>> ans) {

        if (arr.size() == size) {
            ans.add(new ArrayList<>(arr));
            return;
        }
        for (int i = idx; i < nums.length; i++) {
            arr.add(nums[i]);
            subsetBacktrack(arr, nums, i + 1, size, ans);
            arr.removeLast();
        }
    }

    // 78. Subsets
    public List<List<Integer>> subsetsRecursion(int[] nums) {
        // backtracking approach

        List<List<Integer>> ans = new ArrayList<>();
        for (int size = 0; size <= ans.size(); size++) {
            List<Integer> arr = new ArrayList<>();
            subsetBacktrack(arr, nums, 0, size, ans);
        }
        return ans;
    }


    private class CombiSumState {
        public int idx;
        public List<Integer> arr;
        public int total;

        CombiSumState(int idx, List<Integer> arr, int total) {
            this.idx = idx;
            this.arr = arr;
            this.total = total;
        }
    }

    // 39. Combination Sum
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Stack<CombiSumState> stack = new Stack<>();
        stack.push(new CombiSumState(0, new ArrayList<>(), 0));
        Arrays.sort(candidates);
        while (!stack.isEmpty()) {
            CombiSumState curr = stack.pop();
            if (curr.total == target) {
                ans.add(curr.arr);
                continue;
            }

            for (int i = curr.idx; i < candidates.length; i++) {
                int newTotal = curr.total + candidates[i];
                if (newTotal > target) {
                    break;
                }
                List<Integer> newArr = new ArrayList<>(curr.arr);
                newArr.add(candidates[i]);
                stack.push(new CombiSumState(i, newArr, newTotal));
            }
        }
        return ans;
    }

    private void combiBacktrack(int[] candidates, List<Integer> arr, List<List<Integer>> ans,
            int idx, int total, int target) {
        if (total == target) {
            ans.add(new ArrayList<>(arr));
            return;
        }
        for (int i = idx; i < candidates.length; i++) {
            int newTotal = total + candidates[i];
            if (newTotal > target) {
                return;
            }
            arr.add(candidates[i]);
            combiBacktrack(candidates, arr, ans, i, newTotal, target);
            arr.removeLast();
        }
    }

    // 39. Combination Sum
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        combiBacktrack(candidates, new ArrayList<>(), ans, 0, 0, target);
        return ans;
    }

    private class PermuteEntry {
        public Set<Integer> set;
        public List<Integer> arr;

        PermuteEntry(List<Integer> arr, Set<Integer> set) {
            this.arr = arr;
            this.set = set;
        }
    }

    // 46. Permutations
    public List<List<Integer>> permute(int[] nums) {
        // iterative approach
        Stack<PermuteEntry> stack = new Stack<>();
        List<List<Integer>> ans = new ArrayList<>();
        stack.push(new PermuteEntry(new ArrayList<>(), new HashSet<>()));

        while (!stack.isEmpty()) {
            PermuteEntry entry = stack.pop();
            if (entry.arr.size() == nums.length) {
                ans.add(entry.arr);
                continue;
            }
            for (int i = 0; i < nums.length; i++) {
                if (entry.set.contains(i)) {
                    continue;
                }
                List<Integer> newEntry = new ArrayList<>(entry.arr);
                Set<Integer> nSet = new HashSet<>(entry.set);
                newEntry.add(nums[i]);
                nSet.add(i);
                stack.push(new PermuteEntry(newEntry, nSet));
            }
        }
        return ans;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void permuteBacktrack(int[] arr, List<List<Integer>> ans, int pos) {

        if (pos == arr.length) {
            ans.add(Arrays.stream(arr).boxed().toList());
            return;
        }
        for (int i = pos; i < arr.length; i++) {
            swap(arr, pos, i);
            permuteBacktrack(arr, ans, pos + 1);
            swap(arr, pos, i);
        }
    }

    public List<List<Integer>> permute2(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        permuteBacktrack(nums, ans, 0);
        return ans;
    }


    // 79. Word Search
    char[][] board;
    int COL;
    int ROW;
    String WORD;

    public boolean exist(char[][] board, String word) {
        this.COL = board[0].length;
        this.ROW = board.length;
        this.board = board;
        this.WORD = word;
        for (int r = 0; r < this.ROW; r++) {
            for (int c = 0; c < this.COL; c++) {
                if (board[r][c] == word.charAt(0)) {
                    boolean found = this.existBacktrack(r, c, 0);
                    if (found)
                        return true;
                }
            }
        }
        return false;
    }

    boolean existBacktrack(int row, int col, int index) {
        if (index == this.WORD.length()) {
            return true;
        }
        if (row == this.ROW || row < 0 || col == this.COL || col < 0
                || this.board[row][col] != this.WORD.charAt(index)) {
            return false;
        }
        this.board[row][col] = '#';
        int[] rowMoves = {0, 1, 0, -1};
        int[] colMoves = {1, 0, -1, 0};
        for (int i = 0; i < 4; i++) {
            boolean check = existBacktrack(row + rowMoves[i], col + colMoves[i], index + 1);
            if (check)
                return true;
        }
        this.board[row][col] = this.WORD.charAt(index);
        return false;
    }

    // 131. Palindrome Partitioning
    List<List<String>> ans_131 = new ArrayList<>();

    public List<List<String>> partition(String s) {
        int start = 0;
        List<String> arr = new ArrayList<>();
        backtrack_131(arr, start, s);
        return ans_131;
    }

    void backtrack_131(List<String> arr, int start, String s) {
        if (start >= s.length()) {
            this.ans_131.add(new ArrayList<>(arr));
        }
        for (int end = start; end < s.length(); end++) {
            if (isPalindrome(s, start, end)) {
                arr.add(s.substring(start, end + 1));
                backtrack_131(arr, end + 1, s);
                arr.remove(arr.size() - 1);
            }
        }
    }

    boolean isPalindrome(String w, int s, int e) {
        while (s < e) {
            if (w.charAt(e--) != w.charAt(s++)) {
                return false;
            }
        }
        return true;
    }

    // 17. letter combinations of a phone number
    Map<Character, String> dmap = Map.of('2', "abc", '3', "def", '4', "ghi", '5', "jkl", '6', "mno",
            '7', "pqrs", '8', "tuv", '9', "wxyz");
    List<String> ans_17 = new ArrayList<>();

    public List<String> letterCombinations(String digits) {
        // backtracking solution
        if (digits.length() == 0)
            return ans_17;
        StringBuilder sb = new StringBuilder();
        backtrack_17(sb, digits, 0);
        return ans_17;
    }

    void backtrack_17(StringBuilder sb, String digits, int index) {
        if (index >= digits.length()) {
            this.ans_17.add(sb.toString());
            return;
        }
        String digitCombs = dmap.get(digits.charAt(index));
        for (int i = 0; i < digitCombs.length(); i++) {
            sb.append(digitCombs.charAt(i));
            backtrack_17(sb, digits, index + 1);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public List<String> letterCombinations2(String digits) {
        // interative solution
        if (digits.length() == 0)
            return new ArrayList<>();
        List<String> ans = new ArrayList<>(List.of(""));
        for (int i = 0; i < digits.length(); i++) {
            String digitCombo = dmap.get(digits.charAt(i));
            List<String> comb = new ArrayList<>();

            for (String s : ans) {
                for (char c : digitCombo.toCharArray()) {
                    comb.add(s + c);
                }
            }
            ans = comb;
        }
        return ans;
    }



}
