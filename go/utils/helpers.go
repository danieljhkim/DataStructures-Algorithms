package utils

import "sort"

/*
max := -math.MaxFloat64
min := math.MaxFloat64
const MaxInt32 = 1<<31 - 1
const MinInt32 = -1 << 31

const MaxInt64 = 1<<63 - 1
const MinInt64 = -1 << 63
*/


func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sortString(s string) string {
	runes := []rune(s)
	sort.Slice(runes, func(i, j int) bool {
		return runes[i] < runes[j]
	})
	return string(runes)
}

func sortArrayExample() {
	nums := []int{5, 3, 8, 1, 2}

	sort.Ints(nums) // ascending

	// Descending order
	sort.Sort(sort.Reverse(sort.IntSlice(nums)))

    sort.Slice(nums, func(i, j int) bool {
        return nums[i] > nums[j] // descending
    })
}

func sortStructExample() {
    people := []struct {
        Name string
        Age  int
    }{
        {"Alice", 30},
        {"Bob", 25},
        {"Eve", 35},
    }
    
    sort.Slice(people, func(i, j int) bool {
        return people[i].Age < people[j].Age
    })
    
}