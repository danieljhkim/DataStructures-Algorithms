"""
Prime is the building blocks of all numbers
"""


def countPrimes(n: int) -> int:
    if n < 2:
        return 0
    nums = [True] * (n + 1)
    nums[0] = False
    nums[1] = False

    for i in range(2, int(n**0.5) + 1):
        if nums[i]:
            for j in range(i * i, n + 1, i):
                nums[j] = False
    return sum(nums)


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def is_prime(number):
    """Miller-Rabin primality test"""
    if number <= 1:
        return False
    if number == 2 or number == 7 or number == 61:
        return True
    if number % 2 == 0:
        return False

    odd_factor = number - 1
    while odd_factor % 2 == 0:
        odd_factor //= 2

    bases = [2, 7, 61]
    for base in bases:
        exponent = odd_factor
        test_value = pow(base, exponent, number)
        while exponent != number - 1 and test_value != 1 and test_value != number - 1:
            test_value = test_value * test_value % number
            exponent <<= 1
        if test_value != number - 1 and exponent % 2 == 0:
            return False
    return True


def get_all_primes(n):  # sieve_of_eratosthenes
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False  # 0 and 1 are not prime

    p = 2
    while p * p <= n:
        if primes[p]:
            for i in range(p * p, n + 1, p):
                primes[i] = False
        p += 1

    return [i for i, is_prime in enumerate(primes) if is_prime]
