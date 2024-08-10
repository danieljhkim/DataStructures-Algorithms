

"""
Given an integer, find the largest integer, following the rules below:

1. You can swap two adjacent digits, as long as two integers are either both even or both odd.
2. You can swap as many times as you want.
3. Return the largest integer you can get.

"""

def largest_integer(num):
    # Time complexity: O(n^2)
    # Space complexity: O(n)
    digits = list(str(num))
    n = len(digits)

    def can_swap(i, j):
        return (int(digits[i]) % 2 == int(digits[j]) % 2)
    
    swapped = True
    while swapped:
        swapped = False
        for i in range(n - 1):
            if can_swap(i, i + 1) and digits[i] < digits[i + 1]:
                digits[i], digits[i + 1] = digits[i + 1], digits[i]
                swapped = True
    return int(''.join(digits))


def largestInteger(num: int) -> int:
    # Time complexity: O(nlogn)
    # Space complexity: O(n)
    digits = [int(d) for d in str(num)]
    n = len(digits)
    i = 0
    while i < n:
        start = i
        while i < n - 1 and (digits[i] % 2 == digits[i + 1] % 2):
            i += 1
        digits[start:i+1] = sorted(digits[start:i+1], reverse=True)
        i += 1
    return int("".join(map(str, digits)))

