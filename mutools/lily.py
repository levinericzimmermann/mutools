import abjad
import bisect
import functools
import itertools
import operator

try:
    import quicktions as fractions
except ImportError:
    import fractions

import crosstrainer

"""This module contains small functions that may help to generate scores with abjad.

The main focus are functions that may help to convert algorithmically generated abstract
musical data to a form thats closer to notation (where suddenly questions about
time signature, beams, ties and accidentals are occuring).
"""


def mk_no_time_signature():
    return abjad.LilyPondCommand("override Score.TimeSignature.stencil = ##f", "before")


def mk_numeric_ts() -> abjad.LilyPondCommand:
    return abjad.LilyPondCommand(
        "numericTimeSignature", "before"
    )


def mk_staff(voices, clef="percussion") -> abjad.Staff:
    staff = abjad.Staff([])
    for v in voices:
        staff.append(v)
    clef = abjad.Clef(clef)
    abjad.attach(clef, staff)
    abjad.attach(mk_numeric_ts(), staff[0][0])
    return staff


def mk_cadenza():
    return abjad.LilyPondCommand("cadenzaOn", "before")


def mk_bar_line() -> abjad.LilyPondCommand:
    return abjad.LilyPondCommand('bar "|"', "after")


def seperate_by_grid(delay, start, stop, absolute_grid, grid, grid_object) -> tuple:
    grid_start = bisect.bisect_right(absolute_grid, start) - 1
    grid_stop = bisect.bisect_right(absolute_grid, stop)
    passed_groups = tuple(range(grid_start, grid_stop, 1))

    # are_both_on_line = all((start in absolute_grid, stop in absolute_grid))

    if len(passed_groups) == 1:
        return (delay,)

    else:
        delays = []
        is_connectable_per_delay = []
        for i, group in enumerate(passed_groups):
            if i == 0:
                diff = start - absolute_grid[group]
                new_delay = grid[group] - diff
                is_connectable = any(
                    (start in absolute_grid, new_delay.denominator <= 4)
                )
            elif i == len(passed_groups) - 1:
                new_delay = stop - absolute_grid[group]
                is_connectable = any(
                    (stop in absolute_grid, new_delay.denominator <= 4)
                )
            else:
                new_delay = grid[group]
                is_connectable = True

            if new_delay > 0:
                delays.append(new_delay)
                is_connectable_per_delay.append(is_connectable)

        ldelays = len(delays)
        if ldelays == 1:
            return tuple(delays)

        connectable_range = [int(not is_connectable_per_delay[0]), ldelays]
        if not is_connectable_per_delay[-1]:
            connectable_range[-1] -= 1

        solutions = [((n,), m) for n, m in enumerate(delays)]
        for item0 in range(*connectable_range):
            for item1 in range(item0 + 1, connectable_range[-1] + 1):
                connected = sum(delays[item0:item1])
                if abjad.Duration(connected).is_assignable:
                    sol = (tuple(range(item0, item1)), connected)
                    if sol not in solutions:
                        solutions.append(sol)

        possibilites = crosstrainer.Stack(fitness="min")
        amount_connectable_items = len(delays)
        has_found = False
        lsolrange = tuple(range(len(solutions)))
        for combsize in range(1, amount_connectable_items + 1):
            for comb in itertools.combinations(lsolrange, combsize):
                pos = tuple(solutions[idx] for idx in comb)
                items = functools.reduce(operator.add, tuple(p[0] for p in pos))
                litems = len(items)
                is_unique = litems == len(set(items))
                if is_unique and litems == amount_connectable_items:
                    sorted_pos = tuple(
                        item[1] for item in sorted(pos, key=lambda x: x[0][0])
                    )
                    fitness = len(sorted_pos)
                    possibilites.append(sorted_pos, fitness)
                    has_found = True
            if has_found:
                break

        result = possibilites.best[0]
        return tuple(result)


def seperate_by_assignablity(duration, grid) -> tuple:
    def find_sum_in_numbers(numbers, solution, allowed_combinations) -> tuple:
        return tuple(
            sorted(c, reverse=True)
            for c in itertools.combinations_with_replacement(
                numbers, allowed_combinations
            )
            if sum(c) == solution
        )

    if abjad.Duration(duration).is_assignable is True:
        return (abjad.Duration(duration),)

    numerator = duration.numerator
    denominator = duration.denominator
    possible_durations = [
        i
        for i in range(1, numerator + 1)
        if abjad.Duration(i, denominator).is_assignable is True
    ]
    solution = []
    c = 1
    while len(solution) == 0:
        if c == len(possible_durations):
            print(numerator, denominator)
            raise ValueError("Can't find any possible combination")
        s = find_sum_in_numbers(possible_durations, numerator, c)
        if s:
            solution.extend(s)
        else:
            c += 1
    hof = crosstrainer.Stack(size=1, fitness="max")
    for s in solution:
        avg_diff = sum(abs(a - b) for a, b in itertools.combinations(s, 2))
        hof.append(s, avg_diff)
    best = hof.best[0]
    return tuple(fractions.Fraction(s, denominator) for s in best)


def apply_beams(notes, durations, absolute_grid) -> tuple:
    duration_positions = []
    absolute_durations = tuple(itertools.accumulate([0] + list(durations)))
    for dur in absolute_durations:
        pos = bisect.bisect_right(absolute_grid, dur) - 1
        duration_positions.append(pos)
    beam_indices = []
    current = None
    for idx, pos in enumerate(duration_positions):
        if pos != current:
            beam_indices.append(idx)
            current = pos
    for idx0, idx1 in zip(beam_indices, beam_indices[1:]):
        if idx1 == beam_indices[-1]:
            subnotes = notes[idx0:]
        else:
            subnotes = notes[idx0:idx1]
        add_beams = False
        for n in subnotes:
            if type(n) != abjad.Rest:
                add_beams = True
                break
        if len(subnotes) < 2:
            add_beams = False
        if add_beams is True:
            abjad.attach(abjad.Beam(), subnotes)
    return notes


def convert_abjad_pitches_and_mu_rhythms2abjad_notes(
    harmonies: list, delays: list, grid
) -> list:
    leading_pulses = grid.leading_pulses
    absolute_leading_pulses = tuple(
        itertools.accumulate([0] + list(leading_pulses))
    )
    converted_delays = grid.apply_delay(delays)
    absolute_delays = tuple(itertools.accumulate([0] + list(converted_delays)))
    # 1. generate notes
    notes = abjad.Measure(abjad.TimeSignature(grid.absolute_meter), [])
    resulting_durations = []
    for harmony, delay, start, end in zip(
        harmonies, converted_delays, absolute_delays, absolute_delays[1:]
    ):
        subnotes = abjad.Voice()
        seperated_by_grid = seperate_by_grid(
            delay, start, end, absolute_leading_pulses, leading_pulses, grid
        )
        assert sum(seperated_by_grid) == delay
        for d in seperated_by_grid:
            seperated_by_assignable = seperate_by_assignablity(d, grid)
            assert sum(seperated_by_assignable) == d
            for assignable in seperated_by_assignable:
                resulting_durations.append(assignable)
                if harmony:
                    chord = abjad.Chord(harmony, abjad.Duration(assignable))
                else:
                    chord = abjad.Rest(abjad.Duration(assignable))
                subnotes.append(chord)
        if len(subnotes) > 1 and len(harmony) > 0:
            abjad.attach(abjad.Tie(), subnotes[:])
        notes.extend(subnotes)
    assert sum(resulting_durations) == sum(converted_delays)
    # 2. apply beams
    return apply_beams(
        notes, resulting_durations, absolute_leading_pulses
    )
