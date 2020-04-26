from mu.sco import old
from mu.mel import ji

import abc
import itertools
import functools
import operator


class AbstractInterlock(abc.ABC):
    def __init__(self, amount_voices):
        self.amount_voices = amount_voices
        self.mk_pattern()

    def apply(self, melody) -> tuple:
        length_melody = len(melody)
        distribution = []
        while len(distribution) < length_melody:
            try:
                distr = next(self.pattern)
            except StopIteration:
                self.mk_pattern()
                distr = next(self.pattern)
            distribution.append(distr)
        melodies = [[type(melody[0])(None, 0, melody[0].duration)]
                    for i in range(self.amount_voices)]
        pitchchange_cylce = itertools.cycle(
                [0, 0, 0, 1, 0, 1, 0, 2, 0, 0])
        d = melody[0].delay
        first = True
        for tone, distr in zip(melody, distribution):
            hits = 1
            for i, dis in enumerate(distr):
                if dis == 1:
                    next_pitchcycle = next(pitchchange_cylce)
                    if next_pitchcycle == 0:
                        newpitch = tone.pitch.set_val_border(1)
                        newpitch += ji.r(hits, 1)
                    elif next_pitchcycle == 1:
                        newpitch = tone.pitch.copy()
                    else:
                        newpitch = tone.pitch.set_val_border(2).scalar(hits)
                        newpitch = newpitch.set_val_border(1).normalize(2)
                        newpitch = newpitch + ji.r(2, 1)
                    if first is True:
                        melodies[i][0].pitch = newpitch
                        first = False
                    else:
                        melodies[i].append(type(tone)(
                            newpitch, d, tone.duration))
                    hits += 1
            d += tone.delay
        for m in melodies:
            for p0, p1 in zip(m, m[1:]):
                p0.delay = p1.delay - p0.delay
            m[-1].delay = m[-1].duration
        return tuple(old.JIMelody(mel) for mel in melodies)

    @abc.abstractmethod
    def mk_pattern(self) -> "generator":
        raise NotImplementedError


class StaticGrayCodeInterlock(AbstractInterlock):
    def __init__(self, amount_voices, length):
        self.length = length
        AbstractInterlock.__init__(self, amount_voices)

    def mk_pattern(self):
        permutations = type(self).graycode(self.amount_voices, 2)
        permutations = tuple(perm for perm in permutations
                             if sum(perm) > 0)
        modulus = len(permutations)
        graycode = type(self).graycode(self.length, modulus)
        graycode = functools.reduce(operator.add, graycode)
        translation = tuple(permutations[idx] for idx in graycode)
        self.pattern = itertools.cycle(translation)

    @staticmethod
    def graycode(length, modulus):
        """
        Returns the n-tuple reverse Gray code mod m.
        https://yetanothermathblog.com/tag/gray-codes/
        """
        n, m = length, modulus
        F = range(m)
        if n == 1:
            return [[i] for i in F]
        L = StaticGrayCodeInterlock.graycode(n - 1, m)
        M = []
        for j in F:
            M = M + [ll + [j] for ll in L]
        k = len(M)
        Mr = [0] * m
        for i in range(m - 1):
            i1 = i * int(k / m)
            i2 = (i + 1) * int(k / m)
            Mr[i] = M[i1:i2]
        Mr[m - 1] = M[(m - 1) * int(k / m):]
        for i in range(m):
            if i % 2 != 0:
                Mr[i].reverse()
        M0 = []
        for i in range(m):
            M0 = M0 + Mr[i]
        return M0
