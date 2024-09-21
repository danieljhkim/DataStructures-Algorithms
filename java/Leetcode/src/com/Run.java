package leetcode.src.com;
import java.util.stream.Collectors;
import java.util.*;

public class Run {

    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public void swap(List<Integer> arr, int i, int j) {
        int temp = arr.get(i);
        arr.add(i, arr.get(j));
        arr.add(j, temp);
    }

    public void backtrack(List<List<Integer>> ans, List<Integer> arr, int pos) {
        if (pos == arr.size()) {
            ans.add(new ArrayList<>(arr));
            return;
        }
        for (int i = pos; i < arr.size(); i++) {
            swap(arr, i, pos);
            backtrack(ans, arr, pos + 1);
            swap(arr, i, pos);
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> arr = Arrays.stream(nums).boxed().collect(Collectors.toList());
        backtrack(ans, arr, 0);
        return ans;
    }
    
}
