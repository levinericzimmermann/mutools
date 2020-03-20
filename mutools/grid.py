import abc
import itertools
import numpy as np
import operator
import quicktions as fractions


from mu.utils import tools


"""
Module for modelling metrical structures with a strong emphasis on
Messiaen-like added rhythms.
"""


class Prime(object):
    def __init__(
        self, number: int, start: float, end: float, interpolation_function=None
    ):
        assert number > 0
        self.__number = int(number)
        self.start = start
        self.end = end
        self.interpolation_function = interpolation_function

    def copy(self):
        return type(self)(
            int(self.__number),
            float(self.start),
            float(self.end),
            self.interpolation_function,
        )

    @property
    def number(self) -> int:
        return int(self.__number)

    def __repr__(self):
        return repr("{0}, [W: {1} -> {2}]".format(self.__number, self.start, self.end))

    def __add__(self, other):
        return other + self.__number

    def __sub__(self, other):
        return -other + self.__number

    def __mul__(self, other):
        return other * self.__number

    def __neg__(self):
        return -self.__number

    def __pos__(self):
        return +self.__number

    def __int__(self):
        return int(self.__number)

    def interpolate(self, size) -> tuple:
        am = size // self.__number
        if self.interpolation_function is None:
            return tuple(np.linspace(self.start, self.end, am, dtype=float))
        else:
            return self.interpolation_function(self.start, self.end, am)

    def weight(self, size):
        interpolation = self.interpolate(size)
        return sum(interpolation) / len(interpolation)


class Attractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, grid_weights) -> tuple:
        """Return factors to multiply with grid.pulses."""
        raise NotImplementedError


class AttractorIndex(Attractor):
    def __init__(self, index, factor: fractions.Fraction):
        self.index = index
        self.factor = fractions.Fraction(factor)

    def __repr__(self):
        return "AttractorIndex: {0}, {1}".format(self.index, self.factor)

    def copy(self):
        return type(self)(int(self.index), fractions.Fraction(self.factor))

    def __call__(self, grid_weights):
        factors = [1 for i in range(len(grid_weights))]
        factors[self.index] = self.factor
        return tuple(factors)


class AttractExtremes(Attractor):
    def __init__(self, amount, factor: fractions.Fraction):
        self.amount = amount
        self.factor = fractions.Fraction(factor)

    def copy(self):
        return type(self)(int(self.amount), fractions.Fraction(self.factor))

    def mk_factors(self, grid_weights, reverse: bool) -> tuple:
        factors = [1 for i in range(len(grid_weights))]
        weight_idx_pairs = [(i, w) for i, w in enumerate(grid_weights)]
        weight_idx_pairs_sorted = sorted(
            weight_idx_pairs, key=operator.itemgetter(1), reverse=reverse
        )
        for i, pair in enumerate(weight_idx_pairs_sorted):
            if i == self.amount:
                break
            else:
                factors[pair[0]] = self.factor
        return tuple(factors)


class AttractBest(AttractExtremes):
    def __call__(self, grid_weights):
        return self.mk_factors(grid_weights, True)


class AttractWorst(AttractExtremes):
    def __call__(self, grid_weights):
        return self.mk_factors(grid_weights, False)


class Grid(object):
    def __init__(
        self,
        size: int,
        pulse_size: fractions.Fraction,
        primes: list,
        attractors: list = [],
        prefered_absolute_meter: tuple = None,
    ):
        assert size > 0
        self.size = size
        self.pulse_size = fractions.Fraction(pulse_size)
        self.__primes = sorted(primes, key=lambda p: p.number, reverse=True)
        self.__attractors = attractors
        self.__abs_meter = prefered_absolute_meter

    def __repr__(self) -> str:
        return "GRID {0}".format(self.size)

    def copy(self):
        return type(self)(
            self.size,
            fractions.Fraction(self.pulse_size),
            [p.copy() for p in self.primes],
            [att.copy() for att in self.__attractors],
        )

    def apply_delay(self, delays):
        try:
            assert sum(delays) == self.size
        except AssertionError:
            print(sum(delays), delays, self.size)
            raise AssertionError()

        pulses = self.pulses
        delays = tuple(int(d) for d in delays)
        delays = tuple(itertools.accumulate((0,) + delays))
        new_delays = []
        for idx0, idx1 in zip(delays, delays[1:]):
            new_delays.append(fractions.Fraction(sum(pulses[idx0:idx1])))
        return new_delays

    def apply_polyline(self, polyline):
        delays = polyline.delay
        polyline.delay = self.apply_delay(delays)
        polyline.dur = polyline.delay
        return polyline

    @property
    def absolute_size(self) -> float:
        """Size after attractors has been applied."""
        return sum(self.pulses)

    @property
    def primes(self) -> tuple:
        return self.__primes

    @primes.setter
    def primes(self, item: tuple) -> None:
        self.__primes = item

    @property
    def attractors(self):
        return self.__attractors

    @property
    def pulses(self) -> tuple:
        pulses = [fractions.Fraction(self.pulse_size) for i in range(self.size)]
        weights = self.weights
        for att in self.attractors:
            factors = att(weights)
            assert len(factors) == self.size
            pulses = [p * f for p, f in zip(pulses, factors)]
        return tuple(pulses)

    @property
    def weights(self):
        weights = [0 for i in range(self.size)]
        for p in self.primes:
            interpolation = p.interpolate(self.size)
            for idx, w in zip(range(0, self.size, p.number), interpolation):
                weights[idx] += w
        return weights

    @property
    def primes_sorted_by_weights(self) -> tuple:
        """From most important to least important."""
        prime_weight_pairs = [(p, p.weight(self.size)) for p in self.primes]
        ig0 = operator.itemgetter(0)
        ig1 = operator.itemgetter(1)
        return tuple(
            ig0(pair) for pair in sorted(prime_weight_pairs, key=ig1, reverse=True)
        )

    @property
    def most_important_prime(self):
        return self.primes_sorted_by_weights[0]

    @property
    def complete_primes(self):
        return [p for p in self.primes if self.size % p.number == 0]

    @property
    def leading_prime(self):
        complete_primes = self.complete_primes
        primes = [p for p in self.primes_sorted_by_weights if p in complete_primes]
        return primes[0]

    @property
    def leading_pulses(self) -> list:
        pulses = self.pulses
        leading_prime = self.leading_prime
        leading_pulses = [
            sum(pulses[idx0:idx1])
            for idx0, idx1 in zip(
                range(0, self.size, leading_prime.number),
                range(leading_prime.number, self.size + 1, leading_prime.number),
            )
        ]
        return leading_pulses

    @property
    def accent(self) -> str:
        weights = self.weights
        maxidx = weights.index(max(weights))
        diff_start = maxidx
        diff_stop = self.size - maxidx
        if diff_start > diff_stop:
            return "end"
        else:
            return "start"

    @property
    def is_balanced(self) -> bool:
        dominant_prime = int(self.leading_prime)
        weights = self.weights
        maxidx = weights.index(max(weights))
        return maxidx % dominant_prime == 0

    @property
    def absolute_meter(self) -> tuple:
        if self.__abs_meter:
            return self.__abs_meter
        else:
            pulses = self.pulses
            lcm = tools.lcm(*tuple(n.denominator for n in pulses))
            size = sum(pulses)
            n = int(lcm * size)
            while all(
                tuple(
                    all((val % 2 == 0, val / 2 > border))
                    for border, val in ((2, lcm), (1, n))
                )
            ):
                n //= 2
                lcm //= 2
            return (n, lcm)

    def difference(self, other):
        tests = (
            self.leading_prime != other.leading_prime,
            self.size != other.size,
            self.accent != other.accent,
        )
        return sum(tests)
