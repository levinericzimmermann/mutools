"""This module implements the alternating-hands algorithms.

Those algorithms has been descriped by Godfried T. Toussaint
in his paper 'Generating “Good” Musical Rhythms Algorithmically'.
"""

import itertools

from mu.utils import tools


def __mirror(pattern: tuple) -> tuple:
    """Inverse every boolean value inside the tuple.

    Helper function for other functions.
    """
    return tuple(False if item else True for item in pattern)


def paradiddle(size: int) -> tuple:
    def convert2right_left(pattern: tuple) -> tuple:
        right = []
        left = []
        for idx, item in enumerate(pattern):
            if item:
                right.append(idx)
            else:
                left.append(idx)
        return tuple(right), tuple(left)

    cycle = itertools.cycle((True, False))
    pattern = list(next(cycle) for n in range(size))
    pattern[-1] = pattern[-2]
    return convert2right_left(tuple(pattern) + __mirror(pattern)), size * 2


def alternating_hands(seed_rhythm: tuple) -> tuple:
    """Distributes seed_rhythm on right and left hand.

    seed_rhythm is expected to be written in relative form.
    """
    n_elements = len(seed_rhythm)
    absolute_rhythm = tools.accumulate_from_zero(seed_rhythm + seed_rhythm)
    cycle = itertools.cycle((True, False))
    distribution = tuple(next(cycle) for n in range(n_elements))
    distribution += __mirror(distribution)
    right, left = [], []
    for idx, dis in enumerate(distribution):
        item = absolute_rhythm[idx]
        if dis:
            right.append(item)
        else:
            left.append(item)
    return (tuple(right), tuple(left)), absolute_rhythm[-1]


def toggle_rhythm(sharp: bool = True, reverse: bool = False) -> tuple:
    """NotImplementedError!

    x
    x o
    x o o
    x o x X
    x o x o o

    1: x
    2:
       x X
       x o
    3: x X o
       x o o
    4: x o x X
       x o o X
    5: x o x X o
       x o o X o
    6: x o x X o X
       x o o X o o
    7: x o x X o X o
       x x o o X o o
    8: x o x o x X o X
       x o x o o X o o
    9: x o x o x X o X o
       x o o x o o X o o
    10:

    x o x o o X o X o o
    """
    raise NotImplementedError
