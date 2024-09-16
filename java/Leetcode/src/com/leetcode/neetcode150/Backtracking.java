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
        stack.push(new PermuteEntry(new ArrayList<>(),  new HashSet<>()));

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
            permuteBacktrack(arr, ans, pos+1);
            swap(arr, pos, i);
        }
    }

    public List<List<Integer>> permute2(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        permuteBacktrack(nums, ans, 0);
        return ans;
    } 

}
