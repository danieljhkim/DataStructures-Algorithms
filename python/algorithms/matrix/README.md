# Matrix

--- 

### 3 Basic Operations

With these 3 basic operations, we can rotate it around as we wish. 

Reverse Rows
```python
def reverse_rows(matrix):
    for i in range(len(matrix)):
        matrix[i].reverse()
```

Reverse Columns
```python
def reverse_col(matrix):
    h = len(matrix)
    w = len(matrix[0])
    for col in range(w):
        for row in range(h // 2):
            matrix[row][col], matrix[h - row - 1][col] = (
                matrix[h - row - 1][col],
                matrix[row][col],
            )
```

Transpose
```python
def transpose(matrix):
    n = len(matrix)
    for row in range(n):
        for col in range(row, n):
            matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
```

----
### Rotate Time

```python
def rotate_90(matrix):
    transpose(matrix)
    reverse_rows(matrix)

def rotate_180(matrix):
    reverse_rows(matrix)
    reverse_col(matrix)

def rotate_270(matrix):
    transpose(matrix)
    reverse_col(matrix)

def rotate_left_90(matrix):
    transpose(matrix)
    reverse_col(matrix)
```