## Backtracking

- backtracking can be defined as a general algorithmic technic that considers searching every possible combination in order to solve a computational problem

- it is a technique for solving problems recursively by trying to build a solution incrementally, one piece at a time, removing those that fial to satisfy the constraints of the problem at any point in time.

### Types of Backtracking Algorithms

- Decision problem: search for a feasible solution
- Optimization problem: search for the best solution
- Enumeration problem: find all feasible solutions

### Pseudocode
'''
void FIND_SOLUTIONS( parameters):

  if (valid solution):
    store the solution
    Return

  for (all choice):
    if (valid choice):
      APPLY (choice)
      FIND_SOLUTIONS (parameters)
      BACKTRACK (remove choice)

  Return
'''

### complexity
- exponential O(k^N)
- Factorial O(N!)