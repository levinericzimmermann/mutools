from mu.mel import mel
from mu.sco import old

from mu.utils import infit
from mu.utils import interpolations


class SoftLineGlissandoMaker(object):
    """Object to make melodic lines more soft / more floating."""

    def __init__(
        self,
        # the higher the number the higher the chance there will be a glissando if
        # possible
        activity_lv: int = 8,
        # 0 for glissando at the beginning, 1 for glissando at the end, 2 for glissando at
        # the beginning and at the end
        glissando_type_generator: infit.InfIt = infit.Cycle(
            (0, 1, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1)
        ),
        # glissando duration in seconds
        minima_glissando_duration: float = 0.125,
        maxima_glissando_duration: float = 0.25,
        # glissando size in cents
        minima_glissando_size: float = 30,
        maxima_glissando_size: float = 120,
        seed: int = 10,
    ):
        self.__glissando_type_generator = glissando_type_generator
        self.__al = infit.ActivityLevel(activity_lv)
        self.__minima_gissando_duration = minima_glissando_duration
        self.__maxima_glissando_duration = maxima_glissando_duration
        self.__minima_gissando_size = minima_glissando_size
        self.__maxima_glissando_size = maxima_glissando_size

        import random

        random.seed(seed)
        self.__random_module = random

    def get_glissando_values(
        self, halved_duration: float, cent_distance: float
    ) -> tuple:
        """Return Glissando size in cents and duration of the glissando in seconds."""

        # TODO(tidy this mess up!)

        if cent_distance > self.__maxima_glissando_size:
            cent_distance = self.__maxima_glissando_size

        cent_center = (cent_distance - self.__minima_gissando_size) / 2
        cent_center += self.__minima_gissando_size
        distance = (cent_center - self.__minima_gissando_size) / 3

        glissando_size = self.__random_module.gauss(cent_center, distance)
        while (
            glissando_size > self.__maxima_glissando_size
            or glissando_size < self.__minima_gissando_size
        ):
            glissando_size = self.__random_module.gauss(cent_center, distance)

        if self.__maxima_glissando_duration > halved_duration:
            max_duration = halved_duration
        else:
            max_duration = self.__maxima_glissando_duration

        duration_center = (max_duration - self.__minima_gissando_duration) / 2
        duration_center += self.__minima_gissando_duration
        duration_distance = (duration_center - self.__minima_gissando_duration) / 3

        glissando_duration = self.__random_module.gauss(
            float(duration_center), float(duration_distance)
        )
        while (
            glissando_duration > self.__maxima_glissando_duration
            or glissando_duration < self.__minima_gissando_duration
        ):
            glissando_duration = self.__random_module.gauss(
                duration_center, duration_distance
            )

        return glissando_size, glissando_duration

    def __call__(self, melody: old.Melody) -> old.Melody:
        new_melody = melody.copy()
        melody_size = len(melody)

        for idx, tone in enumerate(new_melody):
            halved_duration = tone.duration * 0.5

            # only add ornamentation if there isn't any glissando yet
            if (
                not tone.glissando
                and not tone.pitch.is_empty
                and halved_duration > self.__minima_gissando_duration
            ):
                previous = None
                following = None
                previous_distance = None
                following_distance = None

                if idx != 0 and not melody[idx - 1].pitch.is_empty:
                    previous = melody[idx - 1]
                    previous_distance = previous.pitch.cents - tone.pitch.cents

                if idx + 1 != melody_size and not melody[idx + 1].pitch.is_empty:
                    following = melody[idx + 1]
                    following_distance = following.pitch.cents - tone.pitch.cents

                beginning_and_end_glissando = (
                    previous is not None
                    and abs(previous_distance) > self.__minima_gissando_size,
                    following is not None
                    and abs(following_distance) > self.__minima_gissando_size,
                )

                if any(beginning_and_end_glissando):

                    if next(self.__al):
                        if all(beginning_and_end_glissando):
                            glissando_type = next(self.__glissando_type_generator)
                        else:
                            glissando_type = beginning_and_end_glissando.index(True)

                        glissando_type = ((True, False), (False, True), (True, True))[
                            glissando_type
                        ]

                        glissando_line = []
                        is_first = True
                        for is_allowed, distance in zip(
                            glissando_type, (previous_distance, following_distance)
                        ):
                            if is_allowed:
                                data = self.get_glissando_values(
                                    halved_duration, distance
                                )
                                remaining_time = halved_duration - data[1]
                                if is_first:
                                    data = (
                                        old.PitchInterpolation(
                                            data[1], mel.SimplePitch(0, data[0])
                                        ),
                                        old.PitchInterpolation(
                                            remaining_time, mel.SimplePitch(0, 0)
                                        ),
                                    )
                                else:
                                    data = (
                                        old.PitchInterpolation(
                                            remaining_time, mel.SimplePitch(0, 0)
                                        ),
                                        old.PitchInterpolation(
                                            data[1], mel.SimplePitch(0, 0)
                                        ),
                                        old.PitchInterpolation(
                                            0, mel.SimplePitch(0, data[0])
                                        ),
                                    )
                            else:
                                data = [
                                    old.PitchInterpolation(
                                        halved_duration, mel.SimplePitch(0, 0)
                                    )
                                ]
                                if not is_first:
                                    data.append(
                                        old.PitchInterpolation(0, mel.SimplePitch(0, 0))
                                    )

                            glissando_line.extend(data)
                            is_first = False

                        new_melody[idx].glissando = old.GlissandoLine(
                            interpolations.InterpolationLine(glissando_line)
                        )

        return new_melody


class TremoloMaker(object):
    def __init__(
        self,
        add_tremolo_decider: infit.InfIt = infit.ActivityLevel(6),
        tremolo_size_generator_per_tone: infit.MetaCycle = infit.MetaCycle(
            (infit.Addition, (10, 2))
        ),
    ):
        self.__add_tremolo_decider = add_tremolo_decider
        self.__tremolo_size_generator_per_tone = tremolo_size_generator_per_tone

    def __call__(self, melody: old.Melody) -> old.Melody:
        new_melody = old.Melody([])

        for tone in melody:
            if not tone.pitch.is_empty and next(self.__add_tremolo_decider):
                rhythm = tone.delay
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

                for duration in duration_per_attack:
                    new_melody.append(
                        old.Tone(tone.pitch.copy(), duration, volume=tone.volume)
                    )

            else:
                new_melody.append(tone.copy())

        return new_melody
