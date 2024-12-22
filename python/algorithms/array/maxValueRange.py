from typing import Optional, List


def maximumLengthOfRanges(nums: List[int]) -> List[int]:
    """
    - for each element, find range where it is the biggest.
    - assumption: no duplicate elements
    """
    N = len(nums)
    ans = []
    lcache = {}
    rcache = {}

    def checkLeft(i, val):
        if i == -1:
            return i + 1
        if i in lcache:
            if nums[lcache[i]] > val:
                return lcache[i]
        if val < nums[i]:
            res = i + 1
            lcache[i] = res
        else:
            res = checkLeft(i - 1, val)
        return res

    def checkRight(i, val):
        if i == N:
            return i - 1
        if i in rcache:
            if nums[rcache[i]] > val:
                return rcache[i]
        if val < nums[i]:
            res = i - 1
            rcache[i] = res
        else:
            res = checkRight(i + 1, val)
        return res

    for i, n in enumerate(nums):
        left = checkLeft(i, n)
        right = checkRight(i, n)
        ans.append(right - left + 1)
    return ans


def maximumLengthOfRanges(nums: List[int]) -> List[int]:
    stack = []
    N = len(nums)
    right = [0] * N
    left = [0] * N

    for i, n in enumerate(nums):
        while stack and nums[stack[-1]] < n:
            stack.pop()
        if stack:
            left[i] = stack[-1] + 1
        stack.append(i)

    stack.clear()
    for i in range(N - 1, -1, -1):
        n = nums[i]
        while stack and nums[stack[-1]] < n:
            stack.pop()
        if stack:
            right[i] = stack[-1] - 1
        else:
            right[i] = N - 1
        stack.append(i)
    ans = [right[i] - left[i] + 1 for i in range(N)]
    return ans
