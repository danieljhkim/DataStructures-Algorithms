def insertion_sort(arr):
    # 3 2 1
    for j in range(1, len(arr)):
        i = j
        while i > 0 and arr[i] < arr[i - 1]:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1


def bubble_sort(arr):
    changed = True
    i = 0
    while changed and i < len(arr):
        changed = False
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                changed = True
        i += 1


def part_sort(arr, k):  # k smallest elements
    # O(k * n)
    for i in range(k - 1):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
