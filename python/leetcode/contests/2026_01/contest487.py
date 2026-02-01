from typing import *
from collections import deque


class Solution:
    # 3827. Count Monobit Integers
    def countMonobit(self, n: int) -> int:
        cnt = 0
        for i in range(n + 1):
            bi = bin(i)[2:]
            nset = set(bi)
            if len(nset) == 1:
                cnt += 1
        return cnt

    # 3828. Final Element After Subarray Deletions
    def finalElement(self, nums: List[int]) -> int:
        """
        This problem took me some time to get it.
        Essentially, the optimal decision for Alice comes down to picking the bigger end.

        Optimal Choices (for bob & alice):
        1. remove left
        2. remove right

        0 1 5 2 2 7 3
        6 1 5 2 2 7
        """
        if len(nums) == 1:
            return nums[0]
        return max(nums[0], nums[-1])


# 3829. Design Ride Sharing System
class RideSharingSystem:
    def __init__(self):
        self.rdq = deque()
        self.ddq = deque()
        self.cancelled = set()
        self.waiting = set()  # just this set is required

    def addRider(self, riderId: int) -> None:
        if riderId not in self.waiting:
            self.rdq.append(riderId)
            self.cancelled.discard(riderId)
            self.waiting.add(riderId)

    def addDriver(self, driverId: int) -> None:
        self.ddq.append(driverId)

    def matchDriverWithRider(self) -> List[int]:
        if not self.rdq or not self.ddq:
            return [-1, -1]
        rider = -1
        while self.rdq and self.rdq[0] in self.cancelled:
            out = self.rdq.popleft()
            self.cancelled.discard(out)
        if self.rdq:
            rider = self.rdq.popleft()
            self.waiting.discard(rider)
            return [self.ddq.popleft(), rider]
        return [-1, -1]

    def cancelRider(self, riderId: int) -> None:
        if riderId in self.waiting:
            self.cancelled.add(riderId)
            self.waiting.discard(riderId)
