package leetcode

import (
	"fmt"
	"sort"
)


func containsDuplicate(nums []int) bool {
    nset := make(map[int]bool)
    for _,n := range nums {
        _, exists := nset[n]
        if exists {
            return true
        }
        nset[n] = true
    }
    return false
}

func isAnagram(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    wmap := make(map[rune]int)
    for _,r := range s {
        wmap[r] += 1
    }
    for _,r := range t {
        wmap[r] -= 1
        if wmap[r] < 0 {
            return false
        }
    }
    return true
}

func twoSum(nums []int, target int) []int {
    nmap := make(map[int]int)
    for i, n := range nums {
        c := target - n
        if idx, ok := nmap[c]; ok {
            return []int{idx, i}
        }
        nmap[n] = i
    }
    return []int{}
}

func groupAnagrams(strs []string) [][]string {
    sortStr := func(s string) string {
        runes := []rune(s)
        sort.Slice(runes, func(i, j int) bool {
            return runes[i] < runes[j]
        })
        return string(runes)
    }
    wmap := make(map[string][]string)
    for _, s := range strs {
        key := sortStr(s)
        wmap[key] = append(wmap[key], s)
    }
    res := [][]string{}
    for _, val := range wmap {
        res = append(res, val)
    }
    return res
}

func groupAnagrams2(strs []string) [][]string {
    countKey := func(s string) string {
        count := [26]int{}
        for _, ch := range s {
            count[ch-'a']++
        }
        return fmt.Sprint(count)
    }

    groups := make(map[string][]string)
    for _, s := range strs {
        key := countKey(s)
        groups[key] = append(groups[key], s)
    }

    result := make([][]string, 0, len(groups))
    for _, val := range groups {
        result = append(result, val)
    }
    return result
}

func topKFrequent(nums []int, k int) []int {
    narr := [][2]int{}
    cmap := make(map[int]int)
    for _,n := range nums {
        cmap[n] += 1
    }
    for key,val := range cmap {
        narr = append(narr, [2]int{val, key})
    }
    sort.Slice(narr, func(i,j int) bool {
        return narr[j][0] < narr[i][0]
    })
    res := make([]int, k)
    for i := range k {
        res[i] = narr[i][1]
    }
    return res
}

// 20. Valid Parentheses
func isValid(s string) bool {
    stack := []rune{}
    cmap := map[rune]rune{
        ')':  '(',
        ']': '[',
        '}': '{',
    }
    for _,w := range s {
        if _,ok := cmap[w]; !ok {
            stack = append(stack, w)
        } else {
            if len(stack) == 0 {
                return false
            }
            pair := cmap[w]
            last := stack[len(stack) - 1]
            if pair != last {
                return false
            }
            stack = stack[:len(stack) - 1]
        }
    }
    if len(stack) != 0 {
        return false
    }
    return true

}

// 875. Koko Eating Bananas
func minEatingSpeed(piles []int, h int) int {
    check := func(k int) bool {
        time := 0
        for _, p := range piles {
            time += (p + k - 1) / k
        }
        return time <= h
    }
    low := 1
    high := 1<<31 - 1
    for low <= high {
        mid := (high + low) / 2
        if check(mid) {
            high = mid - 1
        } else {
            low = mid + 1
        }
    }
    return low
}