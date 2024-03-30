# Sliding Window

The sliding window algorithm is a technique used for finding subarrays or sublists within a larger array or list that satisfy a certain condition. It's particularly useful for solving problems where you need to find the longest, shortest, or specific subarray that meets certain criteria (e.g., maximum sum, minimum sum, etc.).

## Fixed size Sliding Window
1. Find the size of the window required, say K.
2. Compute the result for 1st window, i.e. include the first K elements of the data structure.
3. Then use a loop to slide the window by 1 and keep computing the result window by window.


## Variable Size Sliding Window
- In this type of sliding window problem, we increase our right pointer one by one till our condition is true.
- At any step if our condition does not match, we shrink the size of our window by increasing left pointer.
- Again, when our condition satisfies, we start increasing the right pointer and follow step 1.
- We follow these steps until we reach to the end of the array.