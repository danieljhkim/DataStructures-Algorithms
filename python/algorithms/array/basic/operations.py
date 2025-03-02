class BasicArrayOperations:

    def right_shift_vals(self, arr, val):
        """_summary_
        shifts all occurences of val to the right, while maintaining original sequence

        Args:
            val (_type_): value to be shifted
        """
        N = len(arr)
        val_idx = 0

        for i in range(N):
            if arr[i] != val:
                arr[val_idx] = arr[i]
                val_idx += 1

        for i in range(val_idx, N):
            arr[i] = val

        return arr
