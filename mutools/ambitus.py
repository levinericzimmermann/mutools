import abc

from mu.mel import ji


class Ambitus(object):
    def __init__(self, border_down: ji.JIPitch, border_up: ji.JIPitch) -> None:
        try:
            assert border_down < border_up
        except AssertionError:
            msg = "The lower border has to be a lower pitch than the upper border!"
            raise ValueError(msg)

        self.__borders = (border_down, border_up)

    def __repr__(self) -> str:
        return "Ambitus({})".format(self.__borders)

    def __iter__(self) -> iter:
        return iter(self.__borders)

    def __getitem__(self, idx: int):
        return self.__borders[idx]

    def range(self) -> ji.JIPitch:
        return self.__borders[1] - self.__borders[0]

    def find_best_voice_leading(self, pitches: tuple) -> tuple:
        return ji.find_best_voice_leading(pitches, self.__borders)


class AmbitusMaker(abc.ABC):
    @abc.abstractmethod
    def __call__(self, n_voices: int) -> tuple:
        raise NotImplementedError


class SymmetricalRanges(AmbitusMaker):
    def __init__(
        self, centre: ji.JIPitch, range: ji.JIPitch, overlap: ji.JIPitch
    ) -> None:
        self.__centre = centre
        self.__range = range
        self.__overlap = overlap
        self.__halved_overlap = ji.JIPitch.from_cents(overlap.cents / 2)

    @property
    def centre(self) -> ji.JIPitch:
        return self.__centre

    @property
    def range(self) -> ji.JIPitch:
        return self.__range

    @property
    def overlap(self) -> ji.JIPitch:
        return self.__overlap

    def find_lower_neighbour(self, ambitus: Ambitus) -> Ambitus:
        high_border = ambitus[0] + self.__halved_overlap
        low_border = high_border - self.range
        return Ambitus(low_border, high_border)

    def find_higher_neighbour(self, ambitus: Ambitus) -> Ambitus:
        low_border = ambitus[1] - self.__halved_overlap
        high_border = low_border + self.range
        return Ambitus(low_border, high_border)

    def __call__(self, n_voices: int, limit_denominator: int = 32) -> tuple:
        if n_voices % 2 == 0:
            upper_border_lower_centre = self.centre + self.__halved_overlap
            lower_border_higher_centre = self.centre - self.__halved_overlap

            lower_centre = Ambitus(
                upper_border_lower_centre - self.range, upper_border_lower_centre
            )
            higher_centre = Ambitus(
                lower_border_higher_centre, lower_border_higher_centre + self.range
            )

            remaining_per_halve = (n_voices - 2) // 2

            lower = [lower_centre]
            higher = [higher_centre]
            for n in range(remaining_per_halve):
                lower.append(self.find_lower_neighbour(lower[-1]))
                higher.append(self.find_higher_neighbour(higher[-1]))

            ambitus_per_voice = list(reversed(lower)) + higher

        else:
            distance_to_edge = ji.JIPitch.from_cents(
                self.range.cents / 2, limit_denominator
            )
            central_ambitus = Ambitus(
                self.centre - distance_to_edge, self.centre + distance_to_edge
            )
            remaining_per_halve = (n_voices - 1) // 2

            ambitus_per_voice = [central_ambitus]
            for n in range(remaining_per_halve):
                ambitus_per_voice.append(
                    self.find_lower_neighbour(ambitus_per_voice[-1])
                )

            ambitus_per_voice = list(reversed(ambitus_per_voice))
            for n in range(remaining_per_halve):
                ambitus_per_voice.append(
                    self.find_higher_neighbour(ambitus_per_voice[-1])
                )

        return tuple(ambitus_per_voice)
