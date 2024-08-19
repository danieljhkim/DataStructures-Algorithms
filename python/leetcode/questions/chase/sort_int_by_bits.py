"""
You are given an integer array arr. 

Sort the integers in the array in ascending order by the number of 1's in their binary representation

and in case of two or more integers have the same number of 1's you have to sort them in ascending order.

Return the array after sorting it.

https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits

"""

from ast import List


def find_bits(num):
    count = 0
    while num:
        count += num & 1
        num >>= 1
    return count


class Solution:

    def count_bits_2(self, num):
        count = 0
        while num > 0:
            remainder = num % 2
            if remainder == 1:
                count += 1
            num = num // 2
        return count
    
    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) //2
            left = arr[:mid]
            right = arr[mid:]

            left = self.merge_sort(left)
            right = self.merge_sort(right)
            i = j = k = 0

            while i < len(left) and j < len(right):
                if left[i][0] < right[j][0]:
                    arr[k] = left[i]
                    i += 1
                elif left[i][0] == right[j][0]:
                    if left[i][1] < right[j][1]:
                        arr[k] = left[i]
                        i += 1
                    else:
                        arr[k] = right[j]
                        j += 1
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
                j += 1
                k += 1
        return arr
                
    
    def sortByBits(self, arr: List[int]) -> List[int]:
        for i in range(len(arr)):
            arr[i] = (self.find_bits(arr[i]), arr[i])
        arr = self.merge_sort(arr)
        for i in range(len(arr)):
            arr[i] = arr[i][1]
        return arr