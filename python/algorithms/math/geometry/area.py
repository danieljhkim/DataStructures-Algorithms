from math import inf
from typing import Optional, List
from collections import defaultdict


def minAreaRectangle(points: List[List[int]]) -> int:
    """ "
    x1, y2  |  x2, y2
    x1, y1  |  x2, y1
    """
    xadj = defaultdict(list)
    for x, y in points:
        xadj[x].append(y)

    ans = inf
    table = {}
    for x in sorted(xadj):
        ys = xadj[x]
        ys.sort()
        for i in range(len(ys)):
            y1 = ys[i]
            for j in range(i + 1, len(ys)):
                y2 = ys[j]
                if (y1, y2) in table:
                    width = x - table[(y1, y2)]
                    height = y2 - y1
                    area = width * height
                    ans = min(area, ans)
                table[(y1, y2)] = x

    if ans != inf:
        return ans
    return 0
