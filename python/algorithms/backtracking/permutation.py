from collections import defaultdict, deque
from typing import List, Tuple, Optional


class Permutation:

    def permutations(self, arr: list):
        result = []

        def backtrack(arr: list, pos: int):
            if pos == len(arr):
                result.append(arr[:])
                return
            for i in range(pos, len(arr)):
                arr[i], arr[pos] = arr[pos], arr[i]
                backtrack(arr, pos + 1)
                arr[i], arr[pos] = arr[pos], arr[i]

        backtrack(arr, 0)
        return result

    def permute_stack(self, arr):
        stack = [(arr, 0)]  # (current list, current index)
        result = []
        while stack:
            curr_list, index = stack.pop()
            if index == len(curr_list):
                result.append(curr_list.copy())
            else:
                for i in range(index, len(curr_list)):
                    curr_list[index], curr_list[i] = curr_list[i], curr_list[index]
                    stack.append((curr_list, index + 1))
                    curr_list[index], curr_list[i] = curr_list[i], curr_list[index]
        return result

    def permutations_of_certain_size(nums: List[int], size: int) -> List[List[int]]:
        result = []

        def backtrack(current: List[int], used: List[bool]):
            if len(current) == size:
                result.append(current[:])
                return
            for i in range(len(nums)):
                if not used[i]:
                    used[i] = True
                    current.append(nums[i])
                    backtrack(current, used)
                    current.pop()
                    used[i] = False

        backtrack([], [False] * len(nums))
        return result
