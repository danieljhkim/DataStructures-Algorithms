class Sequence:
    """
    - subsequence: derived from the original by deleting some without changing the order of the elements.
    """

    def subsequences(self, arr: list):
        result = []

        def backtrack(index, current):
            if index == len(arr):
                result.append(current[:])
                return
            current.append(arr[index])
            backtrack(index + 1, current)
            current.pop()
            backtrack(index + 1, current)

        backtrack(0, [])
        return result
