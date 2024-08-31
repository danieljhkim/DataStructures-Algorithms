a_list = [1, 2, 3]
b_list = [5, 6, 7]


# recusion solution
def permutation(a_list):
    if len(a_list) == 0:
        return a_list
    if len(a_list) == 1:
        return [a_list]
    all_perms = []
    for i in range(len(a_list)):
        element = a_list[i]
        rem_list = a_list[:i] + a_list[i + 1 :]
        perm = permutation(rem_list)
        for p in perm:
            all_perms.append([element] + p)
    return all_perms


# backtracking solution
all_perms = []
a_list = ["1", "2", "3"]


def permute(a_list, pos):
    length = len(a_list)
    if length == pos:
        all_perms.append(a_list)
    else:
        for i in range(pos, length):
            a_list[pos], a_list[i] = a_list[i], a_list[pos]
            permute(a_list, pos + 1)
            a_list[pos], a_list[i] = a_list[i], a_list[pos]  # backtrack


def permute_stack(a_list):
    stack = [(a_list, 0)]  # Stack contains tuples of (current list, current index)
    perms = []

    while stack:
        print(stack)
        curr_list, index = stack.pop()

        if index == len(curr_list):
            perms.append(curr_list.copy())
        else:
            for i in range(index, len(curr_list)):
                # Swap elements at index and i
                curr_list[index], curr_list[i] = curr_list[i], curr_list[index]
                # Push the new permutation with the next index onto the stack
                stack.append((curr_list.copy(), index + 1))
                # Swap back to restore the original list
                curr_list[index], curr_list[i] = curr_list[i], curr_list[index]

    return perms


def permute_iterative(a_list):
    n = len(a_list)
    perms = []
    indices = list(range(n))
    cycles = list(range(n, 0, -1))

    perms.append(a_list.copy())

    while n:
        for i in reversed(range(n)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1 :] + indices[i : i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                perms.append([a_list[k] for k in indices])
                break
        else:
            return perms


# Example usage:
# a_list = [1, 2, 3]
# permutations = permute_stack(a_list)
# for perm in permutations:
#     print(perm)


"""
permute: 1, 2, 3 | pos: 0 | i: 0
permute: 1, 2, 3 | pos: 1 | i: 1
permute: 1, 2, 3 | pos: 2 | i: 2
final:  1, 2, 3 | pos: 3

backtrack: 1, 2, 3 | pos: 2 | i: 2
backtrack: 1, 2, 3 | pos: 1 | i: 1
permute: 1, 3, 2 | pos: 1 | i: 2
permute: 1, 3, 2 | pos: 2 | i: 2
final:  1, 3, 2 | pos: 3

backtrack: 1, 3, 2 | pos: 2 | i: 2
backtrack: 1, 2, 3 | pos: 1 | i: 2
backtrack: 1, 2, 3 | pos: 0 | i: 0
permute: 2, 1, 3 | pos: 0 | i: 1
permute: 2, 1, 3 | pos: 1 | i: 1
permute: 2, 1, 3 | pos: 2 | i: 2
final:  2, 1, 3 | pos: 3

backtrack: 2, 1, 3 | pos: 2 | i: 2
backtrack: 2, 1, 3 | pos: 1 | i: 1
permute: 2, 3, 1 | pos: 1 | i: 2
permute: 2, 3, 1 | pos: 2 | i: 2
final:  2, 3, 1 | pos: 3

backtrack: 2, 3, 1 | pos: 2 | i: 2
backtrack: 2, 1, 3 | pos: 1 | i: 2
backtrack: 1, 2, 3 | pos: 0 | i: 1
permute: 3, 2, 1 | pos: 0 | i: 2
permute: 3, 2, 1 | pos: 1 | i: 1
permute: 3, 2, 1 | pos: 2 | i: 2
final:  3, 2, 1 | pos: 3

backtrack: 3, 2, 1 | pos: 2 | i: 2
backtrack: 3, 2, 1 | pos: 1 | i: 1
permute: 3, 1, 2 | pos: 1 | i: 2
permute: 3, 1, 2 | pos: 2 | i: 2
final:  3, 1, 2 | pos: 3

backtrack: 3, 1, 2 | pos: 2 | i: 2
backtrack: 3, 2, 1 | pos: 1 | i: 2
backtrack: 1, 2, 3 | pos: 0 | i: 2
"""
