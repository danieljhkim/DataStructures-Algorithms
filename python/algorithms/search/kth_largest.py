def partition(arr: list[int], low: int, high: int) -> int:

    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] >= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def kth_largest_element(arr: list[int], position: int) -> int:

    low, high = 0, len(arr) - 1
    while low <= high:
        if low > len(arr) - 1 or high < 0:
            return -1
        pivot_index = partition(arr, low, high)
        if pivot_index == position - 1:
            return arr[pivot_index]
        elif pivot_index > position - 1:
            high = pivot_index - 1
        else:
            low = pivot_index + 1
    return -1
