import functools
import itertools
import operator

import numpy as np

from mu.mel import ji
from mu.sco import old

MAX_HARMONIC = 32


"""This module adds functions and classes for handling common (sub-)harmonics
between 2 pitches.
"""


class CommonHarmonic(object):
    """Class for modelling common (sub-)harmonics between two pitches.

    Common (sub-)harmonics are regarded as authentic if their register
    doesn't have to be changed to occur in both pitches.
    """

    def __init__(
        self,
        pitch: ji.JIPitch,
        order: tuple,
        gender: bool,
        is_authentic: bool,
        max_low: int = -3,
        max_high: int = 3,
        volume_range: int = 12,
        not_authentic_factor: float = 0.75,
    ) -> None:
        self.__pitch = pitch
        self.__order = sum(order) // len(order)
        self.__gender = gender
        self.__is_authentic = is_authentic

        # how much to distiniguish between closer and further partials
        self.volume_range = volume_range

        # lowest and highest octave
        self.max_low = max_low
        self.max_high = max_high
        self.oct_diff = (max_high - max_low) + 1
        self.harmonic2volume = tuple(
            1 / item for item in np.linspace(1, volume_range, MAX_HARMONIC, dtype=float)
        )

        # how much non authentic partials should be quieter
        self.not_authentic_factor = not_authentic_factor

    def __repr__(self) -> str:
        return "CommonHarmonic({} {})".format(self.pitch, self.order)

    def set_pitch2right_octave(self):
        oc = self.pitch.octave + abs(self.max_low)
        oc = (oc % self.oct_diff) + self.max_low
        return self.pitch.register(oc)

    @property
    def pitch(self) -> ji.JIPitch:
        return self.__pitch

    @property
    def order(self) -> int:
        return self.__order

    @property
    def gender(self) -> bool:
        return self.__gender

    @property
    def is_authentic(self) -> bool:
        return self.__is_authentic

    @property
    def volume(self) -> float:
        v = self.harmonic2volume[self.order]
        if not self.is_authentic:
            v *= self.not_authentic_factor
        return v

    def convert2tone(self, delay: float) -> old.Tone:
        return old.Tone(self.set_pitch2right_octave(), delay, delay, volume=self.volume)


def find_common_harmonics(
    p0: ji.JIPitch, p1: ji.JIPitch, gender: bool = True, border: bool = MAX_HARMONIC
) -> tuple:
    """Find all common (sub-)harmonics between two pitches.

    If gender is True the function return common harmonics.
    If gender is False the function return common subharmonics.

    border declares the highest partial that shall be inspected.

    Return tuple containing CommonHarmonic objects.
    """
    if not p0.is_empty and not p1.is_empty:
        harmonics = tuple(ji.r(b + 1, 1) for b in range(border))

        if not gender:
            harmonics = tuple(p.inverse() for p in harmonics)

        harmonics_per_pitch = tuple(tuple(p + h for h in harmonics) for p in (p0, p1))
        authentic_harmonics = list(
            (h, idx0, harmonics_per_pitch[1].index(h), True)
            for idx0, h in enumerate(harmonics_per_pitch[0])
            if h in harmonics_per_pitch[1]
        )
        normalized_authentic_harmonics = tuple(
            h[0].normalize() for h in authentic_harmonics
        )

        normalized_harmonics_per_pitch = tuple(
            tuple(p.normalize() for p in har) for har in harmonics_per_pitch
        )
        octaves_per_harmonic = tuple(
            tuple(p.octave for p in har) for har in harmonics_per_pitch
        )
        unauthentic_harmonics = []
        for har0_idx, har0 in enumerate(normalized_harmonics_per_pitch[0]):
            if har0 not in normalized_authentic_harmonics:
                if har0 in normalized_harmonics_per_pitch[1]:
                    har1_idx = normalized_harmonics_per_pitch[1].index(har0)
                    oc = tuple(
                        octaves[idx]
                        for octaves, idx in zip(
                            octaves_per_harmonic, (har0_idx, har1_idx)
                        )
                    )
                    unauthentic_harmonics.append((har0,) + oc + (False,))

        return tuple(
            CommonHarmonic(h[0], (h[1], h[2]), gender, h[3])
            for h in authentic_harmonics + unauthentic_harmonics
        )

    else:
        return tuple([])


def mk_harmonics_melodies(
    origin_melodies: tuple, n_voices: int = 5, max_harmonic: int = MAX_HARMONIC
) -> tuple:
    """Make polyphonic movement of common (sub-)harmonics from the origin melodies.

    The resulting tuple contains as many melodies as previously declared with the
    n_voices argument.

    The n_voices argument may be helpful for making sure not having too many
    resulting voices what could happen when voices occasionally contain octaves
    or primes.
    """
    poly_per_interlocking = []
    origin_melodies = tuple(m.discard_rests().tie() for m in origin_melodies)
    for comb in itertools.combinations(origin_melodies, 2):
        cadence = old.Polyphon(comb).chordify(add_longer=True)
        harmonics_per_pitch = tuple(
            functools.reduce(
                operator.add,
                tuple(
                    find_common_harmonics(
                        p[0], p[1], gender=gender, border=max_harmonic
                    )
                    for gender in (True, False)
                ),
            )
            if len(p) == 2
            else tuple([])
            for p in tuple(tuple(h) for h in cadence.pitch)
        )
        poly = [[old.Rest(chord.delay) for chord in cadence] for n in range(n_voices)]
        for h_idx, harmonics in enumerate(harmonics_per_pitch):
            for p_idx, p in enumerate(harmonics[:n_voices]):
                poly[p_idx][h_idx] = p.convert2tone(cadence.delay[h_idx])

        poly = [old.Melody(melody) for melody in poly]
        poly_per_interlocking.append(old.Polyphon(poly))

    return tuple(poly_per_interlocking)
