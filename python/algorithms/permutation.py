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


permute(a_list, 0)

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
