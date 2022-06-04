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