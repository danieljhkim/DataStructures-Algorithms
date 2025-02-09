import math


# 3447
def diviable(groups: list, nums):
    """
    - for n in groups, find first element in nums where, n // element == 0
    - i.e. n is divisible by element
    """
    memo = {}
    visited = set()
    big = max(groups)
    for idx, element in enumerate(nums):
        if element in visited:
            continue
        visited.add(element)
        for i in range(0, big + 1, element):
            if i not in memo:
                memo[i] = idx
    ans = []
    for n in groups:
        if n in memo:
            ans.append(memo[n])
        else:
            ans.append(-1)
    return ans


def get_divisors(n):
    """
    divisors[num] = [list of all divisors of num]
    """
    divisors = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            divisors[j].append(i)
    return divisors


def count_sysmmetric_nums(LOW, HIGH):
    count = 0
    for size in range(1, len(str(HIGH)) + 1):
        for digit in range(1, 10):
            sym = int(str(digit) * size)
            if LOW <= sym <= HIGH:
                count += 1
    return count


def count_symmetric_nums2(LOW, HIGH):
    def generate_same_digit_numbers(digit, length):
        return int(str(digit) * length)

    count = 0
    for digit in range(1, 10):  # Digits from 1 to 9
        length = 1
        while True:
            num = generate_same_digit_numbers(digit, length)
            if num > HIGH:
                break
            if num >= LOW:
                count += 1
            length += 1

    return count


def count_symmetric_nums(LOW, HIGH):
    def how_many(num):
        if num <= 10:
            return num - 1
        return 9

    n = HIGH
    count = 0
    while n > 0:
        count += how_many(n + 1)
        n //= 10
    n = LOW
    while n > 0:
        count -= how_many(n)
        n //= 10
    if LOW:
        return count
    return count


def commonFactors(a: int, b: int) -> int:
    """_summary_
    - number of common factors between a and b
    """
    ans = 1
    low = min(a, b)
    high = max(a, b)
    if low * 2 == high:
        ans += 1
        low = math.ceil(low**0.5)
    for i in range(2, low + 1):
        rem1 = a % i
        rem2 = b % i
        if rem1 == 0 and rem2 == 0:
            ans += 1
    return ans


def commonFactors(a: int, b: int) -> int:
    """_summary_
    - number of common factors between a and b
    """

    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    def count_divisors(n):
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    gcd_ab = gcd(a, b)
    return count_divisors(gcd_ab)


def gcd(x, y):
    while y != 0:
        remainder = x % y
        x = y
        y = remainder
    return x


def commonFactors(self, a: int, b: int) -> int:
    def gcd(x, y):
        temp = y
        while y != 0:
            remainder = x % y
            temp = y
            y = remainder
        return temp

    end = gcd(a, b)
    ans = 1
    for i in range(2, int(end**0.5) + 1):
        if end % i == 0:
            ans += 1
    return ans


def gcd3(a, b):
    if a == 0:
        return b
    return gcd3(b % a, a)


def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def find_lcm_of_list(values: list) -> int:
    if not values:
        return 1
    current_lcm = values[0]
    for val in values[1:]:
        current_lcm = lcm(current_lcm, val)
    return current_lcm


def smallestNumber(n: int, t: int) -> int:
    """ ""
    # You are given two integers n and t.
    # Return the smallest number greater than or equal to n
    # such that the product of its digits is divisible by t.
    """
    prod = 1
    temp = n
    while temp > 0:
        prod *= temp % 10
        temp //= 10
    diff = prod % t
    if diff == 0:
        return n

    return n + t - diff
