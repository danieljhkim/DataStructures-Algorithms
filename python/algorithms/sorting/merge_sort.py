class MergeSort:

    def merge_sort(self, arr):
        if len(arr) < 2:
            return arr
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        left = self.merge_sort(left)
        right = self.merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            k += 1
            j += 1
        return arr


if __name__ == "__main__":
    arr = [3, 6, 8, 10, 1, 2, 1]
    merge_sort = MergeSort()
    merge_sort.merge_sort(arr)
