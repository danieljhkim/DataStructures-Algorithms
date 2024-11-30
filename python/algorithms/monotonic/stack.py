class MonotonicStack:

    def next_smaller_element(self, nums: list) -> list:
        stack = []
        ans = [-1] * len(nums)
        for i in range(len(nums)):
            while stack and nums[stack[-1]] > nums[i]:
                idx = stack.pop()
                ans[idx] = nums[i]
            stack.append(i)
        return ans

    def next_greater_element(self, nums: list) -> list:
        stack = []
        ans = [-1] * len(nums)
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                ans[idx] = nums[i]
            stack.append(i)
        return ans

    def count_monotonic(nums):
        as_count = 1
        is_inc = False if nums[0] > nums[1] else True
        for i in range(1, len(nums) - 1):
            if nums[i] > nums[i + 1]:
                if is_inc:
                    as_count += 1
                is_inc = False
            if nums[i] < nums[i + 1]:
                if not is_inc:
                    as_count += 1
                is_inc = True
        return as_count
