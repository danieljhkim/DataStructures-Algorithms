class Primes:
    """_summary_
    - Prime is the building blocks of all numbers
    """

    def countPrimes(self, n: int) -> int:
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
