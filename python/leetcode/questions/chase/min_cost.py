"""
given an array num, find the minimum cost of the array, following the rules below:

1. You can pick any two integers from num, sum them up, and replace them with the sum. 
2. The sum is then added to the cost.
3. Apply the above rules until num contains only one integer or none.
4. Return the minimum cost you can get after applying the above rules.

"""

import heapq

def min_cost(num):
    # Time complexity: o(n) + O(nlogn) = O(nlogn)
    if len(num) <= 1:
        return 0
    # Convert the array into a min-heap
    heapq.heapify(num) # O(n)
    cost = 0
    while len(num) > 1:
        first = heapq.heappop(num) # O(logn)
        second = heapq.heappop(num) # O(logn)
        total = first + second
        cost += total
        heapq.heappush(num, total) # O(logn)
    return cost


def min_cost2(num):
    # Time complexity: O(n^2 logn) = O(n^2)
    if len(num) <= 1:
        return 0
    num.sort(reverse=True)
    cost = 0
    while len(num) > 1:
        first = num.pop()
        second = num.pop()
        total = first + second
        cost += total
        num.append(total) 
        num.sort() # O(nlogn)
    return cost

