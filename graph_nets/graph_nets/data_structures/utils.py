from typing import Set, Callable, List


def default_comparator(a, b) -> bool:
    return a == b


def sets_equal(s1: Set, s2: Set, comparator: Callable[[any, any], bool] = None) -> bool:
    """
    Compares s1 and s2 with each other, returns True if for every element in s1 there is one in s2 such that s1 == s2.
    Instead of (==) a custom comparator function can be passed.
    """

    if len(s1) != len(s2):
        return False

    s1 = s1.copy()
    s2 = s2.copy()

    if comparator is None:
        comparator = default_comparator

    while len(s1):
        x1 = s1.pop()
        found = False

        # search for x1 in s2
        for x2 in s2:
            if comparator(x1, x2):
                s2.remove(x2)
                found = True
                break
        if not found:
            return False

    return True


def lists_equal(l1: List, l2: List, comparator: Callable[[any, any], bool] = None) -> bool:
    if len(l1) != len(l2):
        return False

    if comparator is None:
        comparator = default_comparator

    for i in range(len(l1)):
        if not comparator(l1[i], l2[i]):
            return False

    return True
