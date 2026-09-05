"""
Max-XOR-pair test harness.
Paste your implementation into `candidate(nums)` below, then run:  python3 xor_test_harness.py
Returns max(a ^ b) over all pairs in nums (nums: list of non-negative ints).
"""

import random, itertools, sys


"""_summary_

optimizations
1. we don't need to check every cand, return the one with highest 1 and no other can offer higher

1 0 1 0 0
1 1 0 0 0
1 0 0 1 0 x
0 1 1 1 1

1 1 1 1
1 0 0 0
0 1 1 0
"""


def max_xor_pair(nums):
    if len(set(nums)) < 2:
        return 0
    W = max(nums).bit_length()
    root = {}
    for x in nums:  # your bucket-insert, but branching
        node = root
        for i in range(W - 1, -1, -1):
            node = node.setdefault((x >> i) & 1, {})
    best = 0
    for c in nums:  # your candidate loop, but the partner set
        node, cur = root, 0  # narrows at every bit instead of one break
        for i in range(W - 1, -1, -1):
            bit = (c >> i) & 1
            nxt = 1 - bit if (1 - bit) in node else bit
            cur |= (bit ^ nxt) << i
            node = node[nxt]
        best = max(best, cur)
    return best


def candidate2(nums):
    if not nums:
        return 0
    bnum, snum = max(nums), min(nums)
    nums = set(nums)
    W = bnum.bit_length() if bnum else 1
    buckets = [set() for _ in range(W)]
    for n in nums:
        b = bin(n)
        for i in range(2, len(b)):
            if b[i] == "1":
                buckets[W - len(b) + i].add(n)
    best = 0
    cands = buckets[0]
    seen = set()
    for i in range(len(buckets)):
        overlap = cands.intersection(buckets[i])
        diff = buckets[i].difference(cands)
        if len(overlap) >= 1:  # when cands match
            if len(diff) == 0:
                seen.update(cands - overlap)
                cands = overlap
        if len(diff) > 0:
            pass

    if snum == 0:
        return max(bnum, best)
    return best


# ============================================================
#  PASTE YOUR IMPLEMENTATION HERE
# ============================================================
def candidate(nums):
    if not nums:
        return 0
    bnum, snum = max(nums), min(nums)
    W = bnum.bit_length() if bnum else 1
    buckets = [[] for _ in range(W)]
    nums = list(set(nums))
    for n in nums:
        b = bin(n)
        for i in range(2, len(b)):
            if b[i] == "1":
                buckets[W - len(b) + i].append(n)
    best = 0
    cands = buckets[0]
    cands = set(cands)
    for c in cands:
        found = False
        seen = set()
        for i in range(len(buckets)):
            overlap = False
            if c in buckets[i]:
                overlap = True
            for n in buckets[i]:
                best = max(best, c ^ n)
                if n not in cands and n not in seen and not overlap:
                    found = True
                seen.add(n)
            if found:
                break

    if snum == 0:
        return max(bnum, best)
    return best


# ============================================================
#  GROUND TRUTH  (O(n^2), only used by the tester)
# ============================================================
def brute(nums):
    if len(set(nums)) < 2:
        return 0
    return max(x ^ y for x in nums for y in nums)


# ============================================================
#  HELPERS
# ============================================================
def run(nums):
    """Return (ok, got, want). ok=False on wrong value OR exception."""
    want = brute(nums)
    try:
        got = candidate(list(nums))
    except Exception as e:
        return (False, f"EXC:{type(e).__name__}:{e}", want)
    return (got == want, got, want)


def fails(nums):
    ok, _, _ = run(nums)
    return not ok


def shrink(nums):
    """Greedily drop elements while the case still fails -> minimal counterexample."""
    nums = list(nums)
    changed = True
    while changed:
        changed = False
        for i in range(len(nums)):
            trial = nums[:i] + nums[i + 1 :]
            if len(trial) >= 2 and fails(trial):
                nums = trial
                changed = True
                break
    return nums


# ============================================================
#  1. CURATED TRAP CASES  (each names the trap it springs)
# ============================================================
CURATED = [
    ([2, 3], "all share top bit -> needs cand x cand"),
    ([6, 7], "share top bit, larger"),
    ([4, 5, 6, 7], "whole block shares top bit"),
    ([1, 3], "minimal: one cand one non-cand"),
    ([12, 4, 3], "greedy trap: optimal partner in a lower bucket"),
    ([1, 4, 12], "empty intervening bucket before the optimum"),
    ([1, 5, 12], "same structure, order-sensitive"),
    ([8, 1, 2, 12, 3], "greedy trap with noise"),
    ([0, 4], "zero is the optimal partner"),
    ([5, 0], "zero, reversed"),
    ([0, 1], "zero vs one"),
    ([0], "single zero -> 0"),
    ([7], "single element -> 0"),
    ([], "empty -> 0"),
    ([5, 5, 5], "all duplicates -> 0"),
    ([3, 3, 1], "dup plus a real pair"),
    ([1, 2, 4, 8, 16], "one bit each, sparse"),
    ([2147483647, 0], "max 31-bit vs zero"),
    ([2**31 - 1, 2**31 - 2], "two large, differ in low bit"),
    ([3, 10, 5, 25, 2, 8], "classic example -> 28 (5^25)"),
    ([14, 70, 8, 50, 20, 9], "mixed magnitudes"),
    ([1 << 20, (1 << 20) + 1], "long shared prefix, differ low"),
]


def test_curated():
    bad = 0
    for nums, why in CURATED:
        ok, got, want = run(nums)
        if not ok:
            bad += 1
            print(f"  FAIL {str(nums):26} got {str(got):>10} want {want:<10} | {why}")
    print(f"[curated]      {len(CURATED) - bad}/{len(CURATED)} passed")
    return bad == 0


# ============================================================
#  2. EXHAUSTIVE over small universes
# ============================================================
def test_exhaustive(universe, sizes):
    bad = 0
    mn = None
    for k in sizes:
        for combo in itertools.combinations(range(universe), k):
            if fails(combo):
                bad += 1
                if mn is None or len(combo) < len(mn):
                    mn = combo
    tag = f"{{0..{universe - 1}}} sizes {sizes[0]}-{sizes[-1]}"
    print(f"[exhaustive]   {tag}: {bad} failures" + (f"  min={mn}" if mn else ""))
    return bad == 0


# ============================================================
#  3. STRUCTURED ADVERSARIAL FAMILIES
# ============================================================
def gen_all_share_top(n, W):
    top = 1 << (W - 1)
    return [top | random.randint(0, top - 1) for _ in range(n)]


def gen_shared_prefix(n, W, shared):
    base = ((1 << shared) - 1) << (W - shared)  # high `shared` bits all 1
    low = W - shared
    return [base | random.randint(0, (1 << low) - 1) for _ in range(n)]


def gen_sparse_powers(n, W):
    return [1 << random.randint(0, W - 1) for _ in range(n)]


def gen_with_zeros(n, W):
    a = [random.randint(0, (1 << W) - 1) for _ in range(n - 1)]
    return a + [0]


def gen_near_duplicates(n, W):
    base = random.randint(0, (1 << W) - 1)
    return [base ^ (1 << random.randint(0, W - 1)) for _ in range(n)]


def gen_two_clusters(n, W):
    a = random.randint(0, (1 << W) - 1)
    b = a ^ ((1 << W) - 1)  # bitwise complement
    return [random.choice((a, b)) ^ random.randint(0, 3) for _ in range(n)]


def test_structured(trials=4000):
    fams = [
        ("all-share-top", gen_all_share_top),
        ("shared-prefix", lambda n, W: gen_shared_prefix(n, W, max(1, W // 2))),
        ("sparse-powers", gen_sparse_powers),
        ("with-zeros", gen_with_zeros),
        ("near-duplicates", gen_near_duplicates),
        ("two-clusters", gen_two_clusters),
    ]
    all_ok = True
    for name, gen in fams:
        bad = None
        for _ in range(trials):
            n = random.randint(2, 10)
            W = random.randint(2, 12)
            nums = gen(n, W)
            if fails(nums):
                bad = shrink(nums)
                break
        if bad is None:
            print(f"[structured]   {name:16} {trials} trials clean")
        else:
            all_ok = False
            ok, got, want = run(bad)
            print(f"[structured]   {name:16} FAIL  min={bad}  got {got}  want {want}")
    return all_ok


# ============================================================
#  4. ORDER-DETERMINISM  (catches hash/iteration-order dependence)
# ============================================================
def test_determinism(trials=20000):
    bad = None
    for _ in range(trials):
        nums = [random.randint(0, 255) for _ in range(random.randint(2, 9))]
        try:
            a = candidate(list(nums))
            b = candidate(list(reversed(nums)))
            c = candidate(random.sample(nums, len(nums)))
        except Exception:
            bad = nums
            break
        if not (a == b == c):
            bad = nums
            break
    if bad is None:
        print(f"[determinism]  {trials} trials: order-independent")
        return True
    print(f"[determinism]  FAIL: answer depends on input order -> {bad}")
    return False


# ============================================================
#  5. RANDOM FUZZ with shrinking
# ============================================================
def test_fuzz(trials=300000, maxlen=10, maxval=1023):
    for _ in range(trials):
        nums = [random.randint(0, maxval) for _ in range(random.randint(2, maxlen))]
        if fails(nums):
            mn = shrink(nums)
            ok, got, want = run(mn)
            print(f"[fuzz]         FAIL after shrink: {mn}  got {got}  want {want}")
            return False
    print(f"[fuzz]         {trials} trials clean (vals 0..{maxval}, len<= {maxlen})")
    return True


def test_fuzz_wide(trials=50000):
    # stress full bit-width and scale of values
    for _ in range(trials):
        nums = [random.randint(0, (1 << 31) - 1) for _ in range(random.randint(2, 8))]
        if fails(nums):
            mn = shrink(nums)
            ok, got, want = run(mn)
            print(f"[fuzz-wide]    FAIL after shrink: {mn}  got {got}  want {want}")
            return False
    print(f"[fuzz-wide]    {trials} trials clean (31-bit values)")
    return True


# ============================================================
#  RUNNER
# ============================================================
def main():
    random.seed(0)  # reproducible; change/remove for fresh inputs
    results = [
        test_curated(),
        test_exhaustive(16, [2, 3, 4, 5]),
        test_exhaustive(32, [2, 3, 4]),
        test_structured(),
        test_determinism(),
        test_fuzz(),
        test_fuzz_wide(),
    ]
    print("\n" + ("ALL GREEN" if all(results) else "NOT GREEN -- see failures above"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
