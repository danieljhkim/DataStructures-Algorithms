package com.algorithms;

public class QuickSort {
	// worse case: O(N^2)
	// average case: O(N log(N))
	// best case: O(N log(N))
	
    public static void main(String[] args) {
        
    }
    
    // swap j and i in arr
    public static void swap(int[] arr, int i, int j) {
    	int temp = arr[i];
    	arr[i] = arr[j];
    	arr[j] = temp;
    }
    
    // takes last element as pivot, then places the pivot element at its correct position
    // and places all smaller to left of pivot and all greater to right of pivot
    public static int partition(int[] arr, int low, int high) {
    	// takes last element as pivot
    	int pivot = arr[high];
    	// index of smaller element and indicates the right position of pivot found so far
    	int i = (low-1);
    	for (int j=low; j<high; j++) {
    		// if current element is smaller than the pivot
    		if (arr[j] < pivot) {
    			// swap j and i, and increment index of smaller element
    			i++;
    			swap(arr, i, j);
    		}
    	}
    	// swap pivot in the right place
    	swap(arr, i+1, high);
    	return (i + 1);
    }
    
    // low = starting index
    // high = ending index
    public static void quickSort(int[] arr, int low, int high) {
    	if (low < high) {
    		int partitionIndex = partition(arr, low, high);
    		//sort elements before partition
    		quickSort(arr, low, partitionIndex-1);
    		//sort elements after partition
    		quickSort(arr, partitionIndex + 1, high);
    	}
    }
}
