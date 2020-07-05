"""This file contains constraints to be used with the RhythmicCP class."""

import abc

from mu.utils import infit


class HRconstraint(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        counterpoint,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_last_harmony(nth_harmony: int, n_harmonies: int) -> bool:
        return nth_harmony + 1 == n_harmonies


class HR_forbid_too_empty_harmonies(HRconstraint):
    """Make sure only n-voices can have a rest per chord."""

    def __init__(
        self, max_n_voices_empty: int, ignore_nth_harmonies: tuple = tuple([])
    ) -> None:
        self.__max_n_voices_empty = max_n_voices_empty
        self.__ignore_nth_harmonies = ignore_nth_harmonies

    def __call__(
        self,
        counterpoint,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:
        for harmony_idx, harmony_and_dissonant_pitches in enumerate(concrete_harmonies):

            if harmony_idx not in self.__ignore_nth_harmonies:

                n_empty_pitches = len(
                    tuple(p for p in harmony_and_dissonant_pitches[0] if p.is_empty)
                )
                if n_empty_pitches > self.__max_n_voices_empty:
                    return False

        return True


class HR_nth_voice_contain_all_possible_pitches(HRconstraint):
    def __init__(self, nth_voice: int):
        self.__nth_voice = nth_voice
        self.__available_pitches = None

    def detect_all_pitches_from_abstract_harmonies(self, harmonies: tuple) -> tuple:
        pitches = []

        for harmony in harmonies:
            if self.__nth_voice not in harmony[1]:
                pitch_idx = self.__nth_voice - len(
                    tuple(pidx for pidx in harmony[1] if pidx < self.__nth_voice)
                )
                pitches.append(harmony[0].blueprint[pitch_idx])

        return tuple(pitches)

    def __call__(
        self,
        counterpoint,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:

        if self.is_last_harmony(nth_harmony, n_harmonies):

            if self.__available_pitches is None:
                self.__available_pitches = set(
                    self.detect_all_pitches_from_abstract_harmonies(
                        counterpoint.harmonies
                    )
                )

            used_pitches = set(
                self.detect_all_pitches_from_abstract_harmonies(abstract_harmonies)
            )
            return used_pitches == self.__available_pitches

        else:
            return True


class HR_assert_nth_harmony_is_of_type_n(HRconstraint):
    def __init__(self, nth_harmony: int, abstract_harmony: tuple):
        self.__nth_harmony = nth_harmony
        self.__abstract_harmony = abstract_harmony

    def __call__(
        self,
        counterpoint,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:
        if nth_harmony >= self.__nth_harmony:
            return abstract_harmonies[self.__nth_harmony] == self.__abstract_harmony
        else:
            return True


class APconstraint(abc.ABC):
    @abc.abstractmethod
    def __call__(self, voice_idx: int, data: tuple) -> tuple:
        raise NotImplementedError


class APconstraint_for_nth_voice(APconstraint):
    def __init__(self, nth_voice: int):
        self.__nth_voice = nth_voice

    @property
    def nth_voice(self) -> int:
        return self.__nth_voice


def _AP_only_nth_voice(function):
    def wrapper(*args):

        if args[0].nth_voice == args[1]:
            return function(*args)
        else:
            return args[2]

    return wrapper


class AP_tremolo(APconstraint_for_nth_voice):
    def __init__(
        self,
        nth_voice: int,
        add_tremolo_decider: infit.InfIt = infit.ActivityLevel(6),
        tremolo_size_generator_per_tone: infit.MetaCycle = infit.MetaCycle(
            (infit.Addition, (10, 2))
        ),
        only_on_non_dissonant_pitches: bool = True,
        define_tremolo_tones_as_dissonant: bool = True,
    ):
        super().__init__(nth_voice)
        self.__add_tremolo_decider = add_tremolo_decider
        self.__tremolo_size_generator_per_tone = tremolo_size_generator_per_tone
        self.__only_on_non_dissonant_pitches = only_on_non_dissonant_pitches
        self.__define_tremolo_tones_as_dissonant = define_tremolo_tones_as_dissonant

    @_AP_only_nth_voice
    def __call__(self, voice_idx: int, data: tuple) -> tuple:
        new_data = [[], [], []]

        for pitch, rhythm, is_not_dissonant_pitch in zip(*data):
            test0 = is_not_dissonant_pitch or not self.__only_on_non_dissonant_pitches
            test0 = test0 and not pitch.is_empty
            if test0 is True and next(self.__add_tremolo_decider):
                duration_per_attack = []

                tremolo_size_generator = next(self.__tremolo_size_generator_per_tone)

                while sum(duration_per_attack) < rhythm:
                    duration_per_attack.append(next(tremolo_size_generator))

                if len(duration_per_attack) > 1:
                    duration_per_attack = duration_per_attack[:-1]
                    difference = rhythm - sum(duration_per_attack)
                    duration_per_attack[-1] += difference
                else:
                    difference = sum(duration_per_attack) - rhythm
                    duration_per_attack[-1] -= difference

                is_first = True
                for duration in duration_per_attack:
                    if is_first:
                        is_not_dissonant = is_not_dissonant_pitch
                        is_first = False
                    else:
                        if self.__define_tremolo_tones_as_dissonant:
                            is_not_dissonant = False
                        else:
                            is_not_dissonant = is_not_dissonant_pitch

                    new_data[0].append(pitch)
                    new_data[1].append(duration)
                    new_data[2].append(is_not_dissonant)

            else:
                new_data[0].append(pitch)
                new_data[1].append(rhythm)
                new_data[2].append(is_not_dissonant_pitch)

        return tuple(tuple(d) for d in new_data)
