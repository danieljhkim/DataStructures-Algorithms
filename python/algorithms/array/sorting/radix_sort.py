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

    def customSortString(self, order: str, s: str) -> str:
        sorted = ""
        freq = [0] * 26
        for char in s:
            freq[ord(char) - ord("a")] += 1
        for char in order:
            sorted += char * (freq[ord(char) - ord("a")])
            freq[ord(char) - ord("a")] = 0
        for i in range(26):
            if freq[i] != 0:
                sorted += chr(i + 97) * freq[i]
        return sorted
