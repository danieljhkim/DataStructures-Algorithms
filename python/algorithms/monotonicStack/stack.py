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
