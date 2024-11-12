class CountSort:
    """
    - time = O(n + k)
    - space = O(n + k)
    - stable
    - values must be non-negatives
    - good when k (range of nums) is not huge compared to n
    """

    def count_sort(arr):
        max_val = max(arr)
        min_val = min(arr)
        num_range = max_val - min_val + 1

        counts = [0] * num_range
        for n in arr:
            counts[n - min_val] += 1
        for i in range(1, num_range):
            counts[i] += counts[i - 1]  # to determine idx

        sorted_arr = [0] * len(arr)
        for n in reversed(arr):
            idx = counts[n - min_val] - 1
            counts[n - min_val] -= 1
            sorted_arr[idx] = n

        return sorted_arr
