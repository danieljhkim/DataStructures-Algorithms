

def partition(arr, low, high):
  pivot = arr[high]
  i = low - 1
  for j in range(low, high):
    if arr[j] <= pivot:
      i += 1
      (arr[i], arr[j]) = (arr[j], arr[i])
  i += 1
  (arr[i], arr[high]) = (arr[high], arr[i])


def quick_sort(arr, low, high):
  if low < high:
    partition = partition(arr, low, high)

    quick_sort(arr, low, partition-1)
    quick_sort(arr, partition+1, high)


"""
45837

j=4      
i=4

j=5
i=5

j=8
i=5
48537

"""