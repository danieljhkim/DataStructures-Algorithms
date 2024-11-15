from typing import Optional, List
import heapq


class Intervals:

    def __init__(self):
        pass

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """_summary_
        find min number of intervals to remove to make them non-overlap
        """
        intervals.sort(key=lambda x: x[1])
        ans = 0
        prev = 0
        for i in range(1, len(intervals)):
            cur = intervals[i]
            if intervals[prev][1] > cur[0]:
                ans += 1
            else:
                prev = i
        return ans

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        """_summary_
        max # of overlaps
        """
        heap = []
        intervals.sort(key=lambda x: x[0])
        i = 0
        ans = 0
        while i < len(intervals):
            v = intervals[i]
            begin = v[0]
            end = v[1]
            if heap and heap[0][0] <= begin:
                heapq.heappop(heap)
            heapq.heappush(heap, (end, begin, i))
            ans = max(ans, len(heap))
            i += 1
        return ans

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        stack = [intervals[0]]
        for interv in intervals[1:]:
            if stack[-1][1] >= interv[0]:
                stack[-1][1] = max(interv[1], stack[-1][1])
            else:
                stack.append(interv)
        return stack
