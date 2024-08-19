
import time

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


    # Time complexity: O(nlogn)
def largestInteger(num: int) -> int:
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


def largestInteger_2(num: int) -> int:
    
    def is_even(number):
        return "even" if number % 2 == 0 else "odd" 
    
    def swap(digits, index, is_up):
        original_num = digits[index]
        num_type = is_even(original_num)
        i = index
        if not is_up:
            while i > 0 and num_type == is_even(digits[i-1]) and digits[i-1] <= original_num:
                i -= 1
        else:
            while i < len(digits)-1 and num_type == is_even(digits[i+1]) and digits[i+1] >= original_num:
                i += 1
        digits[i], digits[index] = digits[index], digits[i]
        return i
        
    digits = [int(d) for d in str(num)]
    sorted_digits = sorted(digits, reverse=True)
    n = len(digits)
    done_index = set()
    while len(sorted_digits) > 0:
        low = sorted_digits.pop()
        high = sorted_digits.pop(0) if sorted_digits else None
        for i in range(n):
            if digits[i] == high and i not in done_index:
                idx = swap(digits, i, False)
                done_index.add(idx)
                high = None
            elif digits[i] == low and i not in done_index:
                idx = swap(digits, i, True)
                done_index.add(idx)
                low = None
            if not low and not high:
                break
    return int("".join(map(str, digits)))

num = 46810989784543216579876543

start = time.time()
print(largest_integer(num))
end = time.time()
print(end - start)
start = time.time()
print(largestInteger(num))
end = time.time()
print(end - start)
start = time.time()
print(largestInteger_2(num))
end = time.time()
print(end - start)
        
                    
            

