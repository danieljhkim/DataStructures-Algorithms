var processQueries = function(queries, m) {
	let marr = [];
	let ans = [];
	for(let i=1; i<m+1; i++) {
		marr.push(i);
	}
	for(let i of queries) {
		let index = marr.indexOf(i);
		ans.push(index);
		marr.splice(index, 1);
		marr = [i, ...marr];
	}
	return ans;
};

var pivotArray = function(nums, pivot) {
	let pivots = [];
	let left = []
	let right = []
	nums.forEach(num => {
		if(num == pivot) {
			pivots.push(pivot);
		} else if(num > pivot) {
			right.push(num);
		} else {
			left.push(num);
		}
	})
	return [...left, ...pivots, ...right];
};

var minOperations = function(n) {
	let ar = [];
	let sum = 0;
	for(let i=0; i<n; i++) {
		ar.push(2 * i + 1);
		sum += ar[i];
	}
	let each = sum/n;
	let l = 0;
	let r = n-1;
	let ans = 0;
	while(l < r) {
		ans++;
		ar[l]++;
		ar[r]--; 
		if(ar[l] == each) {
			l++;
		}
		if(ar[r] == each) {
			r--;
		}
	}
	return ans;
};

var countPrefixes = function(words, s) {
    let ans = 0;
		words.forEach(item => {
			if(s.slice(0,item.length)===item) {
				ans++;
			}
		})
		return ans;
};

//https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/
var replaceElements = function(arr) {
    for(let i=0; i<arr.length; i++) {
			if(i === arr.length-1) {
				arr[i] = -1;
				break;
			}
			let max = 0;
			for(let j=i+1; j<arr.length; j++) {
				if(arr[j] > max) max = arr[j];
			}
			arr[i] = max;
		}
		return arr;
};

//https://leetcode.com/problems/count-operations-to-obtain-zero/
var countOperations = function(num1, num2) {
	let found = false;
	let ans = 0;
	if(num2 === 0 || num1 === 0) {
		return ans;
	}
	while(!found) {
		if(num1 >= num2) {
			num1 -= num2;
			ans++;
		} else {
			num2 -= num1;
			ans++;
		}
		if(num2 === 0 || num1 === 0) {
			found = true;
		}
	}
	return ans;
};

//https://leetcode.com/problems/make-two-arrays-equal-by-reversing-sub-arrays/
var canBeEqual = function(target, arr) {
	target.sort((a,b) => a-b);
	arr.sort((a,b) => a-b);
	for(let i=0; i<arr.length; i++) {
		if(arr[i] !== target[i]) return false;
	}
	return true;
};

//https://leetcode.com/problems/percentage-of-letter-in-string/
var percentageLetter = function(s, letter) {
	let count = 0;
	for(let i=0; i< s.length; i++) {
		if(s.charAt(i) === letter) count++;
	}
	if(count === 0) return 0;
	let ans = Math.floor(count/s.length*100);
	return ans;
};

//https://leetcode.com/problems/kth-distinct-string-in-an-array/
var kthDistinct = function(arr, k) {
	for(let i=0; i<arr.length; i++) {
		let last = arr.lastIndexOf(arr[i]);
		let first = arr.indexOf(arr[i]);
		if(last === first) {
			k--;
			if(k === 0) return arr[i];
		}
	}
	return "";
};

var targetIndices = function(nums, target) {
  nums.sort((a,b) => a-b);
	const ans = [];
	nums.forEach((item, index) => {
		if(item === target) ans.push(index);
	})
	return ans;
};

var reversePrefix = function(word, ch) {
	let rev = [];
	let isFirst = false;
	for(let i=0; i< word.length; i++) {
		rev.push(word.charAt(i));
		if(!isFirst && word.charAt(i) == ch) {
			isFirst = true;
			rev = rev.reverse();
		}
	}
	return rev.join("");
};


var findGCD = function(nums) {
	nums.sort((a,b) => a-b);
	const small = nums[0];
	const large = nums[nums.length-1];
	for(let i=small; i>0; i--) {
		let lr = large%i;
		let sr = small%i;
		if(lr === 0 && sr === 0) return i;
	}
};

var checkString = function(s) {
  let hasB = false;
	for(let i=0; i<s.length; i++) {
		if(s.charAt(i) == "b") hasB = true;
		if(hasB && s.charAt(i)=="a") return false;
	}
	return true;
};

//https://leetcode.com/problems/check-if-number-has-equal-digit-count-and-digit-value/
var digitCount = function(num) {
	for(let i=0; i<num.length; i++) {
		let count = 0;
		for(let j=0; j<num.length; j++) {
			if(Number(num.charAt(j)) == i) count++;
			if(count > Number(num.charAt(i))) return false;
		}
		if(count != Number(num.charAt(i))) return false;
	}
	return true;
};

//https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
var removeDuplicates = function(s) {
	let stack = [];
	for(let i=0; i<s.length; i++) {
		if(stack[stack.length-1] === s[i]){
			stack.pop();
		} else {
			stack.push(s[i]);
		}
	}
	return stack.join("");
};



