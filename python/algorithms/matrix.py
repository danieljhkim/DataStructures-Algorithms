matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def reverse_rows(matrix):
    h = len(matrix)
    w = len(matrix[0])
    for row in range(h):
        for col in range(w // 2):
            matrix[row][col], matrix[row][w - col - 1] = (
                matrix[row][w - col - 1],
                matrix[row][col],
            )
    print(matrix)


def reverse_rows_2(matrix):
    for i in range(len(matrix)):
        matrix[i].reverse()
    print(matrix)


def reverse_col(matrix):
    h = len(matrix)
    w = len(matrix[0])
    for col in range(w):
        for row in range(h // 2):
            matrix[row][col], matrix[h - row - 1][col] = (
                matrix[h - row - 1][col],
                matrix[row][col],
            )
    print(matrix)


def transpose(matrix):
    n = len(matrix)
    for row in range(n):
        for col in range(row, n):
            matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
    print(matrix)


transpose(matrix)
