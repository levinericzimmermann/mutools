import abc


class HRconstraint(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
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
    """Make sure only n voices can have a rest per chord."""

    def __init__(self, max_n_voices_empty: int, ignore_nth_harmonies: tuple) -> None:
        self.__max_n_voices_empty = max_n_voices_empty
        self.__ignore_nth_harmonies = ignore_nth_harmonies

    def __call__(
        self,
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


class HR_contain_all_pitches_of_nth_voice(HRconstraint):
    def __init__(self, voice_idx: int):
        self.__voice_idx = voice_idx

    def __call__(
        self,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:
        if self.is_last_harmony(nth_harmony, n_harmonies):
            # ig = operator.itemgetter(voice_idx)
            return False

        else:
            return True


class HR_assert_nth_harmony_is_of_type_n(HRconstraint):
    def __call__(
        self,
        abstract_harmonies: tuple,
        concrete_harmonies: tuple,
        rhythmic_weight: float,
        duration: int,
        nth_harmony: int,
        n_harmonies: int,
    ) -> bool:
        pass
