class RadixSort:

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
