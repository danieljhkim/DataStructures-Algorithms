

def binary_search(arr, target):
  low = 0
  high = len(arr) - 1

  while low <= high:
    mid = (high-low) // 2 + low
    if target < arr[mid]:
      high = mid - 1
    elif target > arr[mid]:
      low = mid + 1
    else:
      return mid
  return -1


def recursive_search(arr, target, low, high):

  if low <= high:
    mid = (high - low) // 2 + low
    if target == arr[mid]:
      return mid
    elif target < arr[mid]:
      high = mid - 1
      return recursive_search(arr, target, low, high)
    else:
      low = mid + 1
      return recursive_search(arr, target, low, high)
  else:
    return -1

def recursive_search_2(arr, target):

  if len(arr) == 1:
    if arr[0] == target:
      return 0
    else:
      return -1
  else:
    mid = (len(arr) - 1) // 2
    if target == arr[mid]:
      return mid
    elif target < arr[mid]:
      high = mid
      return recursive_search_2(arr[:high], target)
    else:
      low = mid + 1
      return_index = recursive_search_2(arr[low:], target)
      if return_index >= 0:
        return mid + return_index + 1
      else:
        return return_index



a = [1,2,3,4,5,6,7,8,9,10]

print(recursive_search_2(a, 11))