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