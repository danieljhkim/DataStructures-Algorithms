# 3563. Lexicographically Smallest String After Adjacent Removals
# dp
class Solution:
    """_summary_
    You are given a string s consisting of lowercase English letters.

    You can perform the following operation any number of times (including zero):
    - Remove any pair of adjacent characters in the string that are consecutive in the alphabet, in either order
    - Shift the remaining characters to the left to fill the gap.

    Return the lexicographically smallest string that can be obtained after performing the operations optimally.

    Note: Consider the alphabet as circular, thus 'a' and 'z' are consecutive.

    """

    def lexicographicallySmallestString(self, s: str) -> str:  # TLE
        res, N = str(s), len(s)
        chars = {w: ord(w) for w in s}

        def backtrack(arr, idx):
            if idx == N:
                nonlocal res
                w2 = "".join(arr)
                if res > w2:
                    res = w2
                return
            if arr:
                diff = abs(ord(arr[-1]) - chars[s[idx]])
                if diff == 1 or diff == 25:
                    out = arr.pop()
                    backtrack(arr, idx + 1)
                    arr.append(out)
            arr.append(s[idx])
            backtrack(arr, idx + 1)
            arr.pop()

        backtrack([], 0)
        return "".join(res)
