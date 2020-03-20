import itertools
import numpy as np

from mu.mel import ji


class Leaf(object):
    def __init__(self, array, tree):
        self.__array = array
        self.__tree = tree
        self.__hash = Leaf.mk_hash(array)

    @staticmethod
    def mk_hash(array):
        return hash(tuple(array))

    @property
    def array(self):
        return self.__array

    @property
    def tree(self):
        return self.__tree

    def __repr__(self):
        return repr(self.array)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return self.__hash

    def up(self):
        raise NotImplementedError

    def down(self):
        def generator():
            rules_to_apply = np.where(self.array > 0)[0]
            for idx in rules_to_apply:
                for rule in self.tree.rules[idx]:
                    new = Leaf(self.array + rule, self.tree)
                    yield new

        return generator()


class Tree(object):
    def __init__(
        self,
        primes,
        maxlevel,
        compound=True,
        compound_asymmetrical=False,
        three_compound=False,
        allow_stacking=False,
        gender=0,
        octaves=None,
        normalize=2,
        additional_intervals=[],
    ):

        if octaves is not None:
            valid_combinations = self.find_octave_independent_combinations(
                primes,
                maxlevel,
                gender,
                compound,
                compound_asymmetrical,
                three_compound,
                allow_stacking,
                octaves,
                additional_intervals=additional_intervals,
            )
        else:
            valid_combinations = self.find_normalize_dependent_combinations(
                primes,
                maxlevel,
                gender,
                compound,
                compound_asymmetrical,
                three_compound,
                allow_stacking,
                normalize,
                additional_intervals=additional_intervals,
            )
        length_intervals = len(self.intervals)
        self.rules = self.convert_combinations2rules(
            valid_combinations, length_intervals
        )
        objects_lv0 = tuple(
            Leaf(
                np.array(
                    [0 if i != j else 1 for i in range(length_intervals)], dtype=int
                ),
                self,
            )
            for j in range(length_intervals)
        )
        self.__objects_lv0 = objects_lv0

    def convert2pitch_vector(self, array):
        return tuple(
            self.intervals[idx] for idx, num in enumerate(array) for i in range(num)
        )

    def find_normalize_dependent_combinations(
        self,
        primes,
        maxlevel,
        gender,
        compound,
        compound_asymmetrical,
        three_compound,
        allow_stacking,
        normalize,
        additional_intervals,
    ):
        # find all intervals:
        intervals = ji.JIHarmony(
            Tree.find_intervals(
                primes,
                maxlevel,
                gender,
                compound,
                compound_asymmetrical,
                three_compound,
            )
        )
        for additional in additional_intervals:
            intervals.add(additional)
        intervals = [i.set_val_border(1).normalize(normalize) for i in intervals]
        self.intervals = intervals
        comb_intervals = [
            inter for inter in intervals if inter != ji.r(1, 1, val_border=2)
        ]
        if allow_stacking is True:
            combinations = tuple(
                itertools.combinations_with_replacement(comb_intervals, 2)
            )
        else:
            combinations = tuple(itertools.combinations(comb_intervals, 2))
        valid_combinations = []
        for interval in intervals:
            valid = []
            for comb in combinations:
                if (comb[0] + comb[1]).normalize(normalize) == interval:
                    valid.append(comb)
            valid_combinations.append(valid)
        return valid_combinations

    def find_octave_independent_combinations(
        self,
        primes,
        maxlevel,
        gender,
        compound,
        compound_asymmetrical,
        three_compound,
        allow_stacking,
        octaves,
        additional_intervals,
    ):
        # find all intervals:
        intervals = ji.JIHarmony(
            Tree.find_intervals(
                primes,
                maxlevel,
                gender,
                compound,
                compound_asymmetrical,
                three_compound=three_compound,
            )
        )
        for additional in additional_intervals:
            intervals.add(additional)
        comb_intervals = [
            inter for inter in intervals if inter != ji.r(1, 1, val_border=2)
        ]
        if allow_stacking is True:
            combinations = tuple(
                itertools.combinations_with_replacement(comb_intervals, 2)
            )
        else:
            combinations = tuple(itertools.combinations(comb_intervals, 2))
        valid_combinations = []
        for interval in intervals:
            valid = []
            for comb in combinations:
                if comb[0] + comb[1] == interval:
                    valid.append([c.set_val_border(1).normalize(2) for c in comb])
            valid_combinations.append(valid)
        resulting_intervals = [i.set_val_border(1).normalize(2) for i in intervals]
        if octaves is not None:
            add = []
            for octave in octaves:
                for inter in resulting_intervals:
                    add.append(inter + octave)
            resulting_intervals.extend(add)
        resulting_valid_combinations = []
        for interval, valid_combination in zip(
            resulting_intervals, itertools.cycle(valid_combinations)
        ):
            really_valid = []
            for combination in valid_combination:
                for oct_comb in itertools.combinations_with_replacement(octaves, 2):
                    combination0 = [
                        combination[0] + oct_comb[0],
                        combination[1] + oct_comb[1],
                    ]
                    combination1 = [
                        combination[0] + oct_comb[1],
                        combination[1] + oct_comb[0],
                    ]
                    for c in (combination0, combination1):
                        if c[0] + c[1] == interval:
                            really_valid.append(c)
            resulting_valid_combinations.append(really_valid)
        self.intervals = resulting_intervals
        return resulting_valid_combinations

    def convert_combinations2rules(self, valid_combinations, length_intervals):
        rules = []
        for interval_num, combinations in enumerate(valid_combinations):
            r = []
            for comb in combinations:
                idx0 = self.intervals.index(comb[0])
                idx1 = self.intervals.index(comb[1])
                array = []
                for i in range(length_intervals):
                    tests = (i == idx0, i == idx1)
                    if all(tests):
                        array.append(2)
                    elif any(tests):
                        array.append(1)
                    elif i == interval_num:
                        array.append(-1)
                    else:
                        array.append(0)
                array = np.array(array, dtype=int)
                r.append(array)
            rules.append(r)
        return rules

    @staticmethod
    def find_intervals(
        primes,
        maxlevel,
        gender: int,
        compound=True,
        compound_asymmetrical=True,
        three_compound=True,
    ):
        val_border = 2
        intervals = ji.JIHarmony([])
        for p in primes:
            for lv in range(maxlevel):
                scale = lv + 1
                intervals.add(ji.r(p, 1, val_border=val_border).scalar(scale))
                intervals.add(ji.r(1, p, val_border=val_border).scalar(scale))
        if compound is True:
            for p0, p1 in itertools.combinations(primes, 2):
                for lv in range(maxlevel):
                    scale = lv + 1
                    intervals.add(ji.r(p0, p1, val_border=val_border).scalar(scale))
                    intervals.add(ji.r(1, p1 * p0, val_border=val_border).scalar(scale))
                    intervals.add(ji.r(p0 * p1, 1, val_border=val_border).scalar(scale))
                    intervals.add(ji.r(p1, p0, val_border=val_border).scalar(scale))
                if compound_asymmetrical is True:
                    for lv in range(maxlevel):
                        scale0 = lv
                        scale1 = lv + 1
                        p00A = ji.r(p0, 1, val_border=val_border).scalar(scale0)
                        p10A = ji.r(1, p1, val_border=val_border).scalar(scale1)
                        p01A = ji.r(p0, 1, val_border=val_border).scalar(scale1)
                        p11A = ji.r(1, p1, val_border=val_border).scalar(scale0)
                        pres0 = p00A + p10A
                        pres1 = p01A + p11A
                        intervals.add(pres0)
                        intervals.add(pres1)
                        intervals.add(pres0.inverse())
                        intervals.add(pres1.inverse())
        if three_compound is True:
            for p0, p1, p2 in itertools.combinations(primes, 3):
                for lv in range(maxlevel):
                    scale = lv + 2
                    intervals.add(
                        ji.r(p0 + p2, p1, val_border=val_border).scalar(scale)
                    )
                    intervals.add(
                        ji.r(p1 + p2, p0, val_border=val_border).scalar(scale)
                    )
                    intervals.add(
                        ji.r(p0 + p1, p2, val_border=val_border).scalar(scale)
                    )
                if compound_asymmetrical is True:
                    for lv in range(maxlevel):

                        def make_zero_inverse(p0, p1, p2):
                            return p0 + p1 + p2

                        def make_one_inverse(p0, p1, p2):
                            return p0 + p1 + p2.inverse()

                        def make_two_inverse(p0, p1, p2):
                            return p0 + p1.inverse() + p2.inverse()

                        def make_all_inverse(p0, p1, p2):
                            return p0.inverse() + p1.inverse() + p2.inverse()

                        scale0 = lv
                        scale1 = lv + 1
                        scale2 = lv + 2
                        p00A = ji.r(p0, 1, val_border=val_border).scalar(scale0)
                        p10A = ji.r(p1, 1, val_border=val_border).scalar(scale0)
                        p20A = ji.r(p2, 1, val_border=val_border).scalar(scale0)
                        p01A = ji.r(p0, 1, val_border=val_border).scalar(scale1)
                        p11A = ji.r(p1, 1, val_border=val_border).scalar(scale1)
                        p21A = ji.r(p2, 1, val_border=val_border).scalar(scale1)
                        p02A = ji.r(p0, 1, val_border=val_border).scalar(scale2)
                        p12A = ji.r(p1, 1, val_border=val_border).scalar(scale2)
                        p22A = ji.r(p2, 1, val_border=val_border).scalar(scale2)
                        data = [p00A, p10A, p20A, p01A, p11A, p21A, p02A, p12A, p22A]
                        comb = itertools.combinations(data, 3)
                        for c in comb:
                            for p in itertools.permutations(c):
                                intervals.add(make_one_inverse(*p))
                                intervals.add(make_two_inverse(*p))
                                intervals.add(make_all_inverse(*p))

        if gender != 0:
            if gender == 1:
                intervals = (p for p in intervals if p.gender is True)
            elif gender == -1:
                intervals = (p for p in intervals if p.gender is False)
            else:
                raise ValueError("Invalid arg for gender")
        return list(intervals)

    @staticmethod
    def get_nth_permutation(seq, index):
        """Returns the <index>th permutation of <seq>.

        from:
        http://code.activestate.com/recipes/126037-getting-nth-permutation-of-a-sequence/
        """

        seqc = list(seq[:])
        seqn = [seqc.pop()]
        divider = 2  # divider is meant to be len(seqn)+1, just a bit faster
        while seqc:
            index, new_index = index // divider, index % divider
            seqn.insert(new_index, seqc.pop())
            divider += 1
        return seqn

    def get_leaves(self, lv, start=None):
        if start is None:
            leaves = self.__objects_lv0
        else:
            idx = self.intervals.index(start)
            leaves = [self.__objects_lv0[idx]]
        for i in range(lv):
            gen = (l.down() for l in leaves)
            new_leaves = (leaf for generator in gen for leaf in generator)
            leaves = tuple(set(new_leaves))
        return leaves

    def get_converted_leaves(self, lv, start=None):
        leaves = self.get_leaves(lv, start)
        pv = [self.convert2pitch_vector(l.array) for l in leaves]
        return pv
