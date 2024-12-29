from typing import Optional, List
from sortedcontainers import SortedDict

"""Line Sweep Technique

Use Cases:
    - Finding Overlapping Intervals
    - Counting Intervals
    - Finding the Maximum Number of Overlapping Intervals
"""


def count_intervals_covering_points(intervals: List[List[int]], n: int) -> List[int]:
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + 1, -1))

    events.sort()
    result = [0] * (n + 1)
    active_intervals = 0
    j = 0

    for idx in range(n + 1):
        while j < len(events) and events[j][0] <= idx:
            active_intervals += events[j][1]
            j += 1
        result[idx] = active_intervals

    return result


def count_intervals_covering_points(
    intervals: List[List[int]], points: List[int]
) -> List[int]:
    """_summary_
    - good for sparse arrays
    """
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + 1, -1))

    events.sort()
    points.sort()

    result = {}
    active_intervals = 0
    j = 0

    for point in points:
        while j < len(events) and events[j][0] <= point:
            active_intervals += events[j][1]
            j += 1
        result[point] = active_intervals

    return result


def max_overlapping_intervals(intervals, start, end, max_overlap):
    """
    - granular overlapping segments
    """
    table = SortedDict()
    for s, e in intervals:
        table[s] = table.get(s, 0) + 1
        table[e] = table.get(e, 0) - 1

    table[start] = table.get(start, 0) + 1
    table[end] = table.get(end, 0) - 1

    overlaps = 0
    for count in table.values():
        overlaps += count
        if overlaps > max_overlap:
            return False
    return True


if __name__ == "__main__":
    intervals = [[1, 3], [2, 5], [4, 6]]
    points = [1, 2, 3, 4, 5, 6]
    print(count_intervals_covering_points(intervals, points))
    # output {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1}
