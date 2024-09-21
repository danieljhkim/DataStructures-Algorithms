package leetcode.src.com.leetcode;
import java.util.stream.Collectors;
import java.util.*;

public class BacktrackingSolutions {


    Set<List<Integer>> nset = new HashSet<>();

    public void backtrackPermute(List<Integer> arr, int[] nums, boolean[] used) {
        if (arr.size() == nums.length) {
            nset.add(new ArrayList<>(arr));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                arr.add(nums[i]);
                used[i] = true;
                backtrackPermute(arr, nums, used);
                arr.remove(arr.size() - 1);
                used[i] = false;
            }

        }
    }

    // 47. Permutations II
    public List<List<Integer>> permuteUnique(int[] nums) {
        boolean[] used = new boolean[nums.length];
        List<Integer> arr = new ArrayList<>();
        backtrackPermute(arr, nums, used);
        return new ArrayList<>(nset);
    }
    
}
