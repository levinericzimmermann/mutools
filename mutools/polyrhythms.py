import collections
import functools
import operator

from mu.rhy import binr
from mu.utils import prime_factors


class Polyrhythm(object):
    """Class to model rhythmic tuplets.

    Input rhythms are expected to have different size (even though
    no error will be raised if two rhythms have equal size).
    """

    # TODO(think about other ways how an object oriented approach of
    # polyrhythms could be benefical apart from only using the class to
    # transform rhythms of different length to the same absolute length)

    def __init__(self, *rhythm: binr.Compound) -> None:
        duration_per_rhythm = tuple(r.beats for r in rhythm)
        factorised_duration_per_rhythm = tuple(
            collections.Counter(prime_factors.factorise(d)) for d in duration_per_rhythm
        )
        self.__polyrhythmic_identity = self.find_polyrhythmic_identity(
            factorised_duration_per_rhythm
        )
        self.__stretching_factor_per_rhythm = tuple(
            self.find_stretching_factor(factorised_duration, self.polyrhythmic_identity)
            for factorised_duration in factorised_duration_per_rhythm
        )
        self.__transformed_rhythms = tuple(
            rh.real_stretch(factor)
            for rh, factor in zip(rhythm, self.__stretching_factor_per_rhythm)
        )

    @property
    def polyrhythmic_identity(self) -> tuple:
        return self.__polyrhythmic_identity

    @property
    def transformed_rhythms(self) -> tuple:
        return self.__transformed_rhythms

    @staticmethod
    def find_stretching_factor(
        factorised_duration: collections.Counter,
        polyrhythmic_identity: collections.Counter,
    ):
        difference = Polyrhythm.Counter_difference(
            polyrhythmic_identity, factorised_duration
        )
        if difference:
            return functools.reduce(
                operator.mul, tuple(prime ** difference[prime] for prime in difference)
            )
        else:
            return 1

    @staticmethod
    def find_polyrhythmic_identity(factorised_duration_per_rhythm: tuple) -> tuple:
        identity = collections.Counter([])
        for factorised_duration in factorised_duration_per_rhythm:
            difference = Polyrhythm.Counter_difference(factorised_duration, identity)
            for item in difference:
                if difference[item] > 0:
                    identity.update({item: difference[item]})
        return identity

    @staticmethod
    def Counter_difference(
        object0: collections.Counter, object1: collections.Counter
    ) -> collections.Counter:
        """Find difference between two collections.Counter objects.

        Positive numbers if object0 contains more of the item and
        negative numbers if object1 contains more of the item.
        """

        differences = collections.Counter([])
        for item in object0:
            diff = object0[item] - object1[item]
            if diff is not 0:
                differences.update({item: diff})

        for item in object1:
            if object0[item] is 0:
                differences.update({item: -object1[item]})

        return differences
