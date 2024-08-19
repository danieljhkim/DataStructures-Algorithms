"""
given an array num, find the minimum cost of the array, following the rules below:

1. You can pick any two integers from num, sum them up, and replace them with the sum. 
2. The sum is then added to the cost.
3. Apply the above rules until num contains only one integer or none.
4. Return the minimum cost you can get after applying the above rules.

"""

import heapq
import time

def min_cost(num):
    # Time complexity: o(n) + O(nlogn) = O(nlogn)
    if len(num) <= 1:
        return 0
    # Convert the array into a min-heap
    heapq.heapify(num) # O(n)
    cost = 0
    while len(num) > 1: # O(n)
        first = heapq.heappop(num) # O(logn)
        second = heapq.heappop(num) # O(logn)
        total = first + second
        cost += total
        heapq.heappush(num, total) # O(logn)
    return cost


def min_cost2(num):
    # Time complexity: O(nlogn) + O(n^2 nlogn) = O(n^2)
    if len(num) <= 1:
        return 0
    num.sort(reverse=True)
    cost = 0
    while len(num) > 1: # O(n)
        first = num.pop()
        second = num.pop()
        total = first + second
        cost += total
        num.append(total) 
        num.sort(reverse=True) # O(nlogn)
    return cost

def min_cost3(num):
    # Time complexity: O(nLogn) + O(n^2)
    
    def sort_num(num, total): # O(n)
        n = len(num)
        i = 0
        while i < n - 1:
            if num[i] <= total:
                break
            i += 1
        num.insert(i, total)
        return num
    
    if len(num) <= 1:
        return 0
    num.sort(reverse=True) # O(nlogn)
    cost = 0
    while len(num) > 1: # O(n)
        first = num.pop()
        second = num.pop()
        total = first + second
        cost += total
        num = sort_num(num, total) # O(n)
    return cost


num = [1, 2, 3, 4, 5]
start = time.time()
print(min_cost(num.copy()))
end = time.time()
print(end - start)
start = time.time()
print(min_cost2(num.copy()))
end = time.time()
print(end - start)
start = time.time()
print(min_cost3(num.copy()))
end = time.time()
print(end - start)
        
                    
            