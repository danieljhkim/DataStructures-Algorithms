from collections import deque


class QuickSort:

    def _partition(self, arr, low, high):
        pivot = arr[high]
        i = low
        for j in range(low, high):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]
        return i

    def quick_sort_1(self, arr, low, high):
        if low < high:
            partition_index = self._partition(arr, low, high)
            self.quick_sort_1(arr, low, partition_index - 1)
            self.quick_sort_1(arr, partition_index + 1, high)

    def quick_sort1_stack(self, arr):
        stack = deque()
        stack.append((0, len(arr) - 1))
        while stack:
            low, high = stack.pop()
            if low < high:
                pivot_idx = self._partition(arr, low, high)
                stack.append((low, pivot_idx - 1))
                stack.append((pivot_idx + 1, high))
        return arr

    def quick_sort_2(self, arr):
        if len(arr) < 2:
            return arr
        pivot = arr[-1]
        left = []
        right = []
        for i in range(len(arr) - 1):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        left = self.quick_sort_2(left)
        right = self.quick_sort_2(right)
        return left + [pivot] + right

    def quick_sort_2_stack(self, arr):
        stack = deque()
        stack.append(arr)
        sorted_arr = []
        while stack:
            cur = stack.pop()
            if len(cur) > 1:
                pivot = cur[-1]
                left = []
                right = []
                for i in range(len(cur) - 1):
                    if cur[i] < pivot:
                        left.append(cur[i])
                    else:
                        right.append(cur[i])
                stack.append(right + [pivot])
                stack.append(left)
            else:
                sorted_arr.extend(cur)
        return sorted_arr


if __name__ == "__main__":
    arr = [3, 6, 8, 10, 1, 2, 1]
    quick_sort = QuickSort()
    quick_sort.quick_sort_1(arr, 0, len(arr))
