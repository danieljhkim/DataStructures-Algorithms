def max_occupancy(N: int, taken: list, gaps: int):
    """_summary_
    0 0 0 1 0 0 0 1 0 0
    [    ] [     ] [  ]
    """
    if not taken:
        return (N + gaps) // (gaps + 1)
    taken.sort()
    ans = 0
    # 0 to first
    first_gap = taken[0]
    if first_gap > gaps:
        ans += (first_gap) // (gaps + 1)

    for i in range(1, len(taken)):
        prev = taken[i - 1]
        next = taken[i]
        gap = next - prev - 1
        if gap > gaps:
            ans += gap // (gaps + 1)

    last_gap = N - taken[-1] - 1
    if last_gap > gaps:
        ans += last_gap // (gaps + 1)
    return ans
