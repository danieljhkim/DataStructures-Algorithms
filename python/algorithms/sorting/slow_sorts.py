class SlowSorts:

    def insertion_sort(self, arr):
        # 3 2 1
        for j in range(1, len(arr)):
            i = j
            while i > 0 and arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                i -= 1

    def bubble_sort(self, arr):
        changed = True
        i = 0
        while changed and i < len(arr):
            changed = False
            for j in range(0, len(arr) - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    changed = True
            i += 1

    def part_sort(self, arr, k):  # k smallest elements
        # O(k * n)
        for i in range(k - 1):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]

    def redix_sort(self, arr):
        neg_numbers = [-num for num in arr if num < 0]
        pos_numbers = [num for num in arr if num >= 0]
        pos_numbers = self.radix_sort_positive(pos_numbers)
        neg_numbers = self.radix_sort_positive(neg_numbers)
        neg_numbers = [-num for num in reversed(neg_numbers)]
        return neg_numbers + pos_numbers

    def radix_sort_postive(self, arr):
        # positive nums only
        max_digits = len(str(max(arr)))
        for digit in range(max_digits):
            buckets = [[] for _ in range(10)]
            for num in arr:
                cur_digit = (num // 10**digit) % 10
                buckets[cur_digit].append(num)
            i = 0
            for bucket in buckets:
                for n in bucket:
                    arr[i] = n
                    i += 1
        return arr
