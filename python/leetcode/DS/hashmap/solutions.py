class Solution:

    # 1062
    def longestRepeatingSubstring(self, s: str) -> int:
        seen = set()
        n = len(s)
        ans = 0
        for i in range(n):
            j = i + ans
            stop = j - i
            while j <= n and stop <= n // 2:
                if j - i > ans:
                    subtr = s[i:j]
                    if subtr not in seen:
                        seen.add(subtr)
                    else:
                        ans = max(j - i, ans)
                j += 1
        return ans
