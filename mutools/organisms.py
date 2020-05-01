"""organisms contain algorithms for an organic (stochastic) generation of musical data."""

from mu.mel import mel
from mu.mel import ji

from mu.sco import old
from mu.utils import interpolations
from mu.utils import tools

import functools
import itertools
import operator


class Organism(old.Polyphon):
    """The Organism object generates polyphonic structures based on prohability models.

    Each voice that shall be generated has to be decribed by abstract attributes.
    """

    _GRIDSIZE = 1000

    def __init__(
        self,
        action_per_voice: tuple,
        sound_per_voice: tuple,
        allowed_pitches_per_voice: tuple,
        allowed_attacks_per_voice: tuple,
        weights_per_beat: tuple,
        predefined_voices: tuple = tuple([]),
        allow_unisono: bool = True,
        allow_melodic_octaves: bool = False,
        harmonic_weight: float = 2,
        melodic_weight: float = 1,
        random_seed: int = 1000,
        harmonic_range: tuple = (0, 1),
        melodic_range: tuple = (0, 1),
        distance_range: tuple = (0, 1),
        metrical_range: tuple = (0, 1),
    ) -> None:
        self.test_arguments_for_equal_size(
            "voice specific arguments",
            (
                action_per_voice,
                sound_per_voice,
                allowed_attacks_per_voice,
                allowed_pitches_per_voice,
            ),
            lambda data: len(data),
        )
        self.test_arguments_for_equal_size(
            "allowed_attacks_per_voice",
            allowed_attacks_per_voice + tuple(vox.delay for vox in predefined_voices),
            lambda rhythm: sum(rhythm),
        )

        # set attributes
        available_pitches = functools.reduce(operator.add, allowed_pitches_per_voice)
        for vox in predefined_voices:
            available_pitches += tuple(p for p in vox.pitch if not p.is_empty)

        import random

        random.seed(random_seed)

        self._average_distance_per_voice = self._detect_average_distance_per_voice(
            action_per_voice, allowed_attacks_per_voice
        )
        self._linspace_for_distance_likelihood = interpolations.Linear()(
            0, 1, self._GRIDSIZE
        )
        self._duration = len(weights_per_beat)
        self._melodic_weight = melodic_weight
        self._harmonic_weight = harmonic_weight
        self._predefined_voices = predefined_voices
        self._predefined_polyphon = old.Polyphon(predefined_voices)
        self._weights_per_beat = tools.scale(weights_per_beat, *metrical_range)
        self._sound_per_voice = sound_per_voice
        self._allowed_pitches_per_voice = allowed_pitches_per_voice
        self._absolute_allowed_attacks_per_voice = tuple(
            tuple(r.convert2absolute()) for r in allowed_attacks_per_voice
        )
        self._random_module = random
        self._n_voices = len(sound_per_voice)
        self._melodicity_dict_per_voice = self._detect_melodicity_dict_per_voice(
            allowed_pitches_per_voice
        )
        self._harmonicity_dict = self._make_harmonicity_dict(available_pitches)
        self._allow_melodic_octaves = allow_melodic_octaves
        self._allow_unisono = allow_unisono

        self._position_and_voices_pairs = self._detect_position_and_voices_pairs(
            allowed_attacks_per_voice
        )

        super().__init__(self._make_voices())

    @staticmethod
    def _detect_melodicity_dict_per_voice(allowed_pitches_per_voice: tuple) -> tuple:
        melodicity_dict_per_voice = []

        for allowed_pitches in allowed_pitches_per_voice:
            melodicity_dict_per_pitch = dict([])
            for pitch0 in allowed_pitches:
                distance_cents_absolute = []
                for pitch1 in allowed_pitches:
                    distance_cents_absolute.append(
                        (abs((pitch0 - pitch1).cents), pitch1)
                    )
                pitches_sorted_by_distance = tuple(
                    map(
                        operator.itemgetter(1),
                        sorted(
                            distance_cents_absolute,
                            key=operator.itemgetter(0),
                            reverse=True,
                        ),
                    )
                )
                n_pitches = len(allowed_pitches) - 1
                melodicity_dict_per_pitch.update(
                    {
                        pitch0: {
                            other_pitch: idx / n_pitches
                            for idx, other_pitch in enumerate(
                                pitches_sorted_by_distance
                            )
                        }
                    }
                )

            melodicity_dict_per_voice.append(melodicity_dict_per_pitch)

        return tuple(melodicity_dict_per_voice)

    @staticmethod
    def _detect_average_distance_per_voice(
        action_per_voice: tuple, allowed_attacks_per_voice: tuple
    ) -> dict:

        avg_distance_per_voice = []
        for action, allowed_attacks in zip(action_per_voice, allowed_attacks_per_voice):
            n_potential_actions = len(allowed_attacks) * action
            avg_distance_per_voice.append(n_potential_actions)

        return tuple(avg_distance_per_voice)

    @staticmethod
    def _make_harmonicity_dict(pitches: tuple) -> dict:
        """Return dictionary with one entry for each appearing pitch.

        Every pitch entry is linked to another dictionary that contains all appearing
        pitches again. The value of those pitch entries is the harmonicity value between
        both pitches.

        To identify the harmonicity between for instance pitch0 and pitch1 the dictionary
        can be identified through:

            harmonicity_between_pitch0_and_pitch1 = harmonicity_dict[pitch0][pitch1]
        """
        hdict = dict([])

        for p0 in pitches:
            sdict = dict([])
            for p1 in pitches:
                if p0 == p1:
                    harmonicity = 1
                else:
                    harmonicity = (p1 - p0).harmonicity_simplified_barlow

                sdict.update({p1: harmonicity})
            hdict.update({p0: sdict})

        return hdict

    @staticmethod
    def _detect_position_and_voices_pairs(allowed_attacks_per_voice: tuple) -> tuple:
        """Return tuple with subtuples. Each subtuple represents one beat.

        Each subtuple contains two elements:
            (1) absolute position of the beat
            (2) voice indices from those voices that potentially could have an attack on
                this position
        """
        harmonic_changes = []
        for position in functools.reduce(
            lambda r0, r1: r0.union(r1), allowed_attacks_per_voice
        ).convert2absolute():
            relevant_voices = []
            for vox_idx, voice in enumerate(allowed_attacks_per_voice):
                if position in voice.convert2absolute():
                    relevant_voices.append(vox_idx)

            assert int(position) == position
            harmonic_changes.append((int(position), tuple(sorted(relevant_voices))))
        return tuple(harmonic_changes)

    def _detect_distance_to_last_action(
        self, vox_idx: int, current_position: int, previous_element: tuple
    ) -> int:
        absolute_attacks = self._absolute_allowed_attacks_per_voice[vox_idx]
        current = absolute_attacks.index(current_position)
        previous = absolute_attacks.index(previous_element[0])
        return current - previous

    def _test_voice(
        self, distance_likelihood: float, harmonicity: float, metricity: float
    ) -> bool:
        """Detect if particular voice shall be activated to generate a new action."""

        likelihood = (distance_likelihood + harmonicity + metricity) / 3
        return self._random_module.random() < likelihood

    def _find_start_harmony(self) -> tuple:
        nth_voice_get_activated = tuple(range(self._n_voices))
        previous_pitches_per_voice = tuple(None for i in range(self._n_voices))
        active_pitches = tuple(vox[0].pitch for vox in self._predefined_voices)
        return self._detect_pitch_per_voice(
            nth_voice_get_activated, previous_pitches_per_voice, active_pitches
        )

    def _detect_absolute_harmonicity(
        self, pitch: ji.JIPitch, other_pitches: tuple
    ) -> float:
        if pitch.is_empty or not other_pitches:
            return 0.5

        else:
            summed_harmonicity = sum(
                self._harmonicity_dict[pitch][other_pitch]
                for other_pitch in other_pitches
                if not other_pitch.is_empty
            )
            return summed_harmonicity / len(other_pitches)

    def _detect_relative_harmonicity(
        self, vox_idx: int, pitch: ji.JIPitch, other_pitches: tuple
    ) -> float:
        harmonicity_per_pitch = {
            p: self._detect_absolute_harmonicity(p, other_pitches)
            for p in self._allowed_pitches_per_voice[vox_idx]
        }
        sorted_harmonicity = tuple(sorted(harmonicity_per_pitch.values()))
        harmonicity_position = sorted_harmonicity.index(harmonicity_per_pitch[pitch])
        return harmonicity_position / (len(harmonicity_per_pitch) - 1)

    def _detect_pitch_per_voice(
        self,
        nth_voice_get_activated: tuple,
        previous_pitch_per_voice: tuple,
        active_pitches: tuple,
    ) -> tuple:

        # (1) figure out whether the voices that are intented to make a new action will
        # either play a new tone or will have a rest
        make_tone_per_voice = []
        for voice_idx, last_pitch in zip(
            nth_voice_get_activated, previous_pitch_per_voice
        ):
            if last_pitch is not None and last_pitch.is_empty:
                make_tone_per_voice.append(True)
            else:
                make_tone_per_voice.append(
                    self._random_module.random() < self._sound_per_voice[voice_idx]
                )

        # (2) make harmonic ranking for all possible combinations
        nth_voice_shall_play_a_tone = tuple(
            vox_idx
            for vox_idx, shall_play_tone in zip(
                nth_voice_get_activated, make_tone_per_voice
            )
            if shall_play_tone
        )
        relevant_previous_pitch_per_voice = tuple(
            p
            for shall_play_tone, p in zip(make_tone_per_voice, previous_pitch_per_voice)
            if shall_play_tone
        )

        if nth_voice_shall_play_a_tone:
            combination_and_harmonicity = []
            for combination in itertools.product(
                *tuple(
                    self._allowed_pitches_per_voice[vox_idx]
                    for vox_idx in nth_voice_shall_play_a_tone
                )
            ):
                # CHECK FIRST IF THERE ARE NO UNISONO
                is_allowed = all(
                    potential_pitch != previous_pitch
                    for potential_pitch, previous_pitch in zip(
                        combination, relevant_previous_pitch_per_voice
                    )
                )

                if not self._allow_unisono and is_allowed:
                    simultan_pitches = tuple(combination) + active_pitches
                    simplified_pitches = set(p.register(0) for p in simultan_pitches)
                    is_allowed = (
                        len(simultan_pitches) - len(set(simplified_pitches)) == 0
                    )

                if not self._allow_melodic_octaves and is_allowed:
                    is_allowed = all(
                        potential_pitch.register(0) != previous_pitch.register(0)
                        for potential_pitch, previous_pitch in zip(
                            combination, relevant_previous_pitch_per_voice
                        )
                        if previous_pitch is not None and not previous_pitch.is_empty
                    )

                if is_allowed:
                    pitch, *other = tuple(combination)
                    harmonicity = self._detect_absolute_harmonicity(
                        pitch, tuple(other) + tuple(active_pitches)
                    )
                    combination_and_harmonicity.append((combination, harmonicity))

            combinations_sorted_by_harmonicity = tuple(
                map(
                    operator.itemgetter(0),
                    sorted(combination_and_harmonicity, key=operator.itemgetter(1)),
                )
            )

            n_combinations = len(combinations_sorted_by_harmonicity) - 1

            # (3) detect fitness per combination (combine melodicity and harmonicity)
            combinations_and_fitness = []
            for comb_idx, combination in enumerate(combinations_sorted_by_harmonicity):
                harmonicity = comb_idx / n_combinations
                melodicity = sum(
                    self._melodicity_dict_per_voice[v_idx][previous_pitch][
                        potential_pitch
                    ]
                    if previous_pitch is not None and not previous_pitch.is_empty
                    else 1
                    for v_idx, previous_pitch, potential_pitch in zip(
                        nth_voice_shall_play_a_tone,
                        relevant_previous_pitch_per_voice,
                        combination,
                    )
                )
                melodicity /= len(combination)
                fitness = (harmonicity * self._harmonic_weight) + (
                    melodicity * self._melodic_weight
                )
                combinations_and_fitness.append((combination, fitness))

            combinations_sorted_by_fitness = tuple(
                map(
                    operator.itemgetter(0),
                    sorted(
                        combination_and_harmonicity,
                        key=operator.itemgetter(1),
                        reverse=True,
                    ),
                )
            )

            choosen_combination = iter(combinations_sorted_by_fitness[0])

        else:
            choosen_combination = iter([])

        return tuple(
            next(choosen_combination) if shall_play_tone else mel.TheEmptyPitch
            for shall_play_tone in make_tone_per_voice
        )

    def _make_voices(self) -> tuple:
        new_voices = [[(0, start_pitch)] for start_pitch in self._find_start_harmony()]

        position_and_voices_pairs_and_next_position = tuple(
            pv_pair + (next_position,)
            for pv_pair, next_position in zip(
                self._position_and_voices_pairs,
                tuple(map(operator.itemgetter(0), self._position_and_voices_pairs))[1:]
                + (self._duration,),
            )
        )

        for (
            position,
            voices,
            next_position,
        ) in position_and_voices_pairs_and_next_position:

            # for position == 0, the _find_start_harmony method generate the relevant data
            if position != 0:

                # (1) detect likelihood regarding metricity
                metricity = self._weights_per_beat[position]

                # (2) detect likelihood regarding distance to last action
                distance_per_voice = tuple(
                    self._detect_distance_to_last_action(
                        vox_idx, position, new_voices[vox_idx][-1]
                    )
                    for vox_idx in voices
                )

                distance_likelihood_per_voice = []
                for vox_idx, distance in zip(voices, distance_per_voice):

                    avg_distance = self._average_distance_per_voice[vox_idx]
                    if avg_distance > distance:
                        distance_percentage = distance / avg_distance
                        distance_percentage *= self._GRIDSIZE
                        distance_likelihood = self._linspace_for_distance_likelihood[
                            int(distance_percentage)
                        ]
                    else:
                        distance_likelihood = 1

                    distance_likelihood_per_voice.append(distance_likelihood)

                # (3) detect likelihood regarding harmonicity (the lower the higher)
                data_in_predefined_voices = self._predefined_polyphon.cut_up_by_time(
                    start=position, stop=next_position, hard_cut=True, add_earlier=True
                )
                previous_pitches_per_voice = tuple(
                    map(operator.itemgetter(1), tuple(vox[-1] for vox in new_voices))
                )
                harmonicity_per_voice = []
                for vox in voices:
                    vox_pitch = previous_pitches_per_voice[vox]
                    if not vox_pitch.is_empty:
                        other_pitches = tuple(
                            p
                            for idx, p in enumerate(previous_pitches_per_voice)
                            if idx != vox
                        )
                        other_pitches += tuple(
                            vox[0].pitch for vox in data_in_predefined_voices
                        )
                        summed_harmonicity = sum(
                            self._harmonicity_dict[vox_pitch][other_pitch]
                            for other_pitch in other_pitches
                            if not other_pitch.is_empty
                        )
                        harmonicity = summed_harmonicity / len(other_pitches)
                    else:
                        harmonicity = 0.5

                    harmonicity_per_voice.append(harmonicity)

                voice_shall_be_activated = tuple(
                    self._test_voice(distance_likelihood, harmonicity, metricity)
                    for harmonicity, distance_likelihood in zip(
                        harmonicity_per_voice, distance_likelihood_per_voice
                    )
                )

                if any(voice_shall_be_activated):
                    nth_voice_shall_be_activated = tuple(
                        voices[idx]
                        for idx in tools.find_all_indices_of_n(
                            True, voice_shall_be_activated
                        )
                    )

                    previous_pitches_per_voice_for_voices_that_shall_be_activated = tuple(
                        previous_pitches_per_voice[v_idx]
                        for v_idx in nth_voice_shall_be_activated
                    )

                    active_pitches = tuple(
                        previous_pitches_per_voice[v_idx]
                        for v_idx in range(self._n_voices)
                        if v_idx not in nth_voice_shall_be_activated
                    )
                    active_pitches += tuple(
                        vox[0].pitch for vox in data_in_predefined_voices
                    )

                    pitches_per_voice = self._detect_pitch_per_voice(
                        nth_voice_shall_be_activated,
                        previous_pitches_per_voice_for_voices_that_shall_be_activated,
                        active_pitches,
                    )
                    for pitch, vox_idx in zip(
                        pitches_per_voice, nth_voice_shall_be_activated
                    ):
                        new_voices[vox_idx].append((position, pitch))

        # generate real Melody and Tone objects
        ig0, ig1 = map(operator.itemgetter, (0, 1))
        for vox_idx, vox in enumerate(new_voices):
            absolute_positions = tuple(map(ig0, vox))
            pitches = tuple(map(ig1, vox))
            vox = old.Melody(
                [
                    old.Tone(pitch, b - a)
                    for pitch, a, b in zip(
                        pitches,
                        absolute_positions,
                        absolute_positions[1:] + (self._duration,),
                    )
                ]
            )
            new_voices[vox_idx] = vox

        # concatenate predefined voices with newly generated voices
        return self._predefined_voices + tuple(new_voices)

    @staticmethod
    def test_arguments_for_equal_size(
        name: str, iterable: tuple, test_function
    ) -> None:
        """Raise error in case any of the elements has an unequal size"""
        size_of_first_item = test_function(iterable[0])
        for idx, size in enumerate(test_function(item) for item in iterable):
            try:
                assert size == size_of_first_item
            except AssertionError:
                msg = "{} {} contain only {} elements while {} ".format(
                    name, idx, size, size_of_first_item
                )
                msg += "elements are expected."
                raise ValueError(msg)
