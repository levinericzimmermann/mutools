"""This module contains different algorithms that model counterpoint-like structures."""

import abc
import bisect
import collections
import functools
import itertools
import operator

from mu.mel import ji
from mu.mel import mel

from mu.rhy import binr

from mu.utils import activity_levels
from mu.utils import infit
from mu.utils import tools

from . import ambitus
from . import counterpoint_constraints as constraints


class Counterpoint(abc.ABC):
    """Abstract superclass for modelling counterpoint.

    Arguments for Counterpoint classes:

        MANDATORY:

        (1) a tuple containing harmonies, where every harmony is composed of:
            [a] BlueprintHarmony (the actual harmony)
            [b] voices to skip (voices that are silent)
            [c] BlueprintHarmony (optional dissonant pitches per harmony)
                => (HarmonyItself, Voices2skip, AdditionalDissonantPitches)
            - every harmony has to contain the same amount of pitches!

        (2) rhythm per voice
            - there has to be as many elements as there are voices
            - use a binr.Compound object for each input
            - every element has to have equal size (sum of durations)
            - depending on the particular subclass this either indicates the
              possible points where a new tone could be played or each rhythm is
              equal with the resulting rhythm of each voice


        OPTIONAL:

        (3) weight per beat
            - has to have equal size like the rhythms per voice
            - a list containing floating point numbers 0 <= x <= 1 (percentage)

        (4) constraints for harmonic resolution
            - functions for the harmonic frame algorithm
            - each function should have as an input:
                * counterpoint: the current counterpoint object
                * harmonies: abstract (tuple)
                * harmonies: concrete (tuple)
                * rhythmic weight (float)
                * duration of the current harmony (int)
                * harmony counter (int)
                * how many harmonies have to be found (int)

        (5) constraints for dissonant added pitches

        (6) step definition in cents. default: (70, 270)

        (7) ambitus_maker. default: make symmetrical ambitus with inner point 1/1,
            range 3/1 and overlap 5/4

        (8) start harmony: the first harmony (abstract form) -> if a solution can be
            found with this harmony
    """

    def __init__(
        self,
        harmonies: tuple,
        possible_attacks_per_voice: tuple,
        weights_per_beat: tuple = None,
        constraints_added_pitches: tuple = tuple([]),
        stepsize: tuple = (70, 270),
        ambitus_maker: ambitus.AmbitusMaker = ambitus.SymmetricalRanges(
            ji.r(1, 1), ji.r(3, 1), ji.r(5, 4)
        ),
        start_harmony: tuple = None,
        add_dissonant_pitches_to_nth_voice: tuple = None,
    ) -> None:

        if add_dissonant_pitches_to_nth_voice is None:
            add_dissonant_pitches_to_nth_voice = tuple(
                True for i in possible_attacks_per_voice
            )

        # testing for correct arguments
        self.test_arguments_for_equal_size(
            "Harmony", harmonies, lambda harmony: sum(len(h) for h in harmony[:2])
        )
        self.test_arguments_for_equal_size(
            "Rhythm", possible_attacks_per_voice, lambda rhythm: sum(rhythm)
        )

        self._duration = possible_attacks_per_voice[0].beats
        self._n_voices = len(possible_attacks_per_voice)

        if weights_per_beat is None:
            weights_per_beat = tuple(1 for i in range(self._duration))

        self._start_harmony = start_harmony
        self._stepsize = stepsize
        self._stepsize_centre = stepsize[1] - stepsize[0]
        self._ambitus_per_voice = ambitus_maker(self._n_voices)
        self._weights_per_beat = weights_per_beat
        self._constraints_added_pitches = constraints_added_pitches
        self._harmonies = harmonies
        self._possible_attacks_per_voice = possible_attacks_per_voice
        self._add_dissonant_pitches_to_nth_voice = add_dissonant_pitches_to_nth_voice

        self._harmonic_network = self.make_harmonic_network(
            self._n_voices, harmonies, harmonies
        )
        self._attack_voice_pairs = self.find_beat_positions(possible_attacks_per_voice)

    @property
    def harmonies(self) -> tuple:
        return self._harmonies

    @staticmethod
    def find_beat_positions(possible_attacks_per_voice: tuple) -> tuple:
        """Find which voice is having a new beat at which position.

        Return tuple filled with tuples. The first element of those subtuples is the
        absolute position in the rhythm and the second element contains the
        indices of all voices who have a beat on this position.
        """
        harmonic_changes = []
        for position in functools.reduce(
            lambda r0, r1: r0.union(r1), possible_attacks_per_voice
        ).convert2absolute():
            relevant_voices = []
            for vox_idx, voice in enumerate(possible_attacks_per_voice):
                if position in voice.convert2absolute():
                    relevant_voices.append(vox_idx)

            assert int(position) == position
            harmonic_changes.append((int(position), tuple(sorted(relevant_voices))))
        return tuple(harmonic_changes)

    @staticmethod
    def make_harmonic_network(n_voices: int, departure: tuple, arrival: tuple) -> dict:
        """Return dict where every harmony has one entry.

        Each entry is linked to a dict where the keys
        are tuples that indicate the voices that will
        change and the values are tuples that
        contain all possible following harmonies for this
        particular change. One harmony that could follo
        to the previous one is composed of two objects
        where the first one is the actually harmony and
        the second one is a tuple that contains a bool for
        each tone whose pitch get changed -> True if the next
        is an actual pitch and False if the next pitch equal
        TheEmptyPitch.

        Input are two tuples, departure and arrival.
        Both contain harmonies. Departure and arrival can be
        equal. Departure contains harmonies where to start with
        and arrival contains harmonies where to go to.
        """

        def simplify_harmonic_notation(harmony: tuple) -> tuple:
            """Simplifies harmonic notation.

            Return tuple that contains one item per voice / pitch.
            One of those items could either be TheEmptyPitch object
            or a tuple containing BlueprintPitch and indices.
            """
            simplified = list(harmony[0].blueprint)
            for n in sorted(harmony[1]):
                simplified.insert(n, mel.TheEmptyPitch)
            return tuple(simplified)

        departure_simplified = tuple(simplify_harmonic_notation(h) for h in departure)
        arrival_simplified = tuple(simplify_harmonic_notation(h) for h in arrival)

        network = {}
        for departing_harmony, departing_harmony_simplified in zip(
            departure, departure_simplified
        ):

            possible_arrivals = {}
            available_voices = tuple(range(n_voices))
            for n in available_voices:
                for com in itertools.combinations(available_voices, n + 1):
                    possible_arrivals.update({com: []})

            for arriving_harmony, arriving_harmony_simplified in zip(
                arrival, arrival_simplified
            ):
                changed_pitches = tuple(
                    idx
                    for idx, p in enumerate(departing_harmony_simplified)
                    if arriving_harmony_simplified[idx] != p
                )

                if changed_pitches:
                    change_type_indicator = tuple(
                        False
                        if arriving_harmony_simplified[idx] is mel.TheEmptyPitch
                        else True
                        for idx in changed_pitches
                    )
                    possible_arrivals[changed_pitches].append(
                        (arriving_harmony, change_type_indicator)
                    )

            network.update(
                {
                    departing_harmony: {
                        key: tuple(possible_arrivals[key]) for key in possible_arrivals
                    }
                }
            )
        return network

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

    @staticmethod
    def convert_abstract_harmonies(primes: tuple, abstract_harmonies: tuple) -> tuple:
        """Convert abstract harmonic notation to real pitches.

        The expected form for input per harmony is the same as it is expected
        when initalizing a RhythmicCP object:

            (BlueprintHarmony,
            list of silent voices,
            BlueprintHarmony for dissonant pitches)

        The resulting form per harmony is tuple containing two elements:

            (Harmony itself,
            additional dissonant pitches)

        where both elements are tuples that contain ji.JIPitch objects.
        """

        converted_harmonies = []
        for harmony in abstract_harmonies:
            harmony_itself = list(harmony[0](*primes))
            for empty_pitch_index in harmony[1]:
                harmony_itself.insert(empty_pitch_index, mel.TheEmptyPitch)
            dissonant_pitches = harmony[2](*primes)
            converted_harmonies.append((tuple(harmony_itself), dissonant_pitches))

        return tuple(converted_harmonies)

    def _extract_harmonic_frame_voices(
        self, converted_harmonic_frame: tuple, rhythm_per_voice: tuple
    ) -> tuple:
        ig0 = operator.itemgetter(0)
        only_harmonies = tuple(
            ig0(harmony_and_additional_dissonant_pitches)
            for harmony_and_additional_dissonant_pitches in converted_harmonic_frame
        )

        pitches_per_voice = []

        for amb, pitches in zip(self._ambitus_per_voice, tuple(zip(*only_harmonies))):

            pitches_without_repetitions = [pitches[0]]

            for pitch in pitches[1:]:
                if pitches_without_repetitions[-1] != pitch:
                    pitches_without_repetitions.append(pitch)

            pitches_per_voice.append(
                amb.find_best_voice_leading(pitches_without_repetitions)
            )

        return tuple(
            (pitches, rhythm)
            for pitches, rhythm in zip(pitches_per_voice, rhythm_per_voice)
        )

    def __call__(self, *primes: int) -> tuple:
        """Solve counterpoint algorithm and convert to musical data.

        The output format is a tuple containing two further tuples.
        Each of those subtuples has one entry per voice.

        In the (1) subtuple one of those entries has the form:
            (PITCHES, RHYTHM, IS_NON_DISSONANT_PITCH)
        with the data types
            (tuple(JIPitch), binr.Compound, tuple(bool))

        In the (2) subtuple one of those entries has the form:
            (PITCHES, RHYTHM)
        with the data types
            (tuple(JIPitch), binr.Compound)

        The voices in the first subtuple have additional dissonant pitches,
        while the voices in the second subtuple only contain the pitches of the
        inner harmonic frame.
        """

        # (1) find harmonic frame
        hf_data = self._find_harmonic_frame(primes)
        harmonic_frame_abstract, rhythm_per_voice, harmonic_index_per_beat = hf_data
        harmonic_frame_converted = self.convert_abstract_harmonies(
            primes, harmonic_frame_abstract
        )

        # (2) extract harmonic frame voices, put pitches in correct octave
        voices_without_added_pitches = self._extract_harmonic_frame_voices(
            harmonic_frame_converted, rhythm_per_voice
        )

        # (3) add additional dissonant pitches (interpolation)
        voices_with_added_pitches = self._add_dissonant_pitches(
            voices_without_added_pitches,
            harmonic_frame_abstract,
            harmonic_frame_converted,
            harmonic_index_per_beat,
        )

        return voices_with_added_pitches, voices_without_added_pitches

    #                                                                           ##
    # methods for the harmonic frame finding backtracking algorithm:            ##
    # (private since they won't be needed outside)                              ##
    #                                                                           ##

    @abc.abstractmethod
    def _find_harmonic_frame(self, primes: tuple) -> tuple:
        """Return tuple that contains two subtuples.

        The first tuple contains one harmony for each change of pitch.
        The second tuple contains rhythms per voice.
        """

        raise NotImplementedError

    #                                                                           ##
    # methods for the dissonant-pitch-adding algorithm:                         ##
    # (private since they won't be needed outside)                              ##
    #                                                                           ##

    def __filter_pitches_that_are_in_right_direction(
        self, origin: ji.JIPitch, candidates: tuple, is_rising: bool
    ) -> tuple:
        """Filter and analyse relationship of candidates towards the origin.

        Return tuple filled with subtuples where each subtuple contains:
            (Pitch, distance_to_origin_in_cents)
        """
        allowed_pitches = []
        for pitch in candidates:
            distance = pitch - origin
            distance_cents = distance.cents
            is_right_direction = any(
                (
                    is_rising and distance_cents > 0,
                    (not is_rising) and distance_cents < 0,
                )
            )

            if is_right_direction:
                allowed_pitches.append((pitch, distance_cents))

        return tuple(allowed_pitches)

    def __filter_pitches_that_are_in_step_distance(self, candidates: tuple) -> tuple:
        allowed_pitches = []
        for pitch_and_cent_pair in candidates:
            abs_ct = abs(pitch_and_cent_pair[1])
            if abs_ct > self._stepsize[0] and abs_ct < self._stepsize[1]:
                allowed_pitches.append(pitch_and_cent_pair)
        return tuple(allowed_pitches)

    @staticmethod
    def __set_pitches_to_correct_frame(
        pitches: tuple, border0: ji.JIPitch, border1: ji.JIPitch
    ) -> tuple:
        if border0 > border1:
            border0, border1 = border1, border0

        octaves = set((border0.octave, border1.octave))

        allowed_pitches = []
        for pitch in pitches:
            for octave in octaves:
                adapted_pitch = pitch.register(octave)
                if adapted_pitch > border0 and adapted_pitch < border1:
                    allowed_pitches.append(adapted_pitch)

        return tuple(allowed_pitches)

    def __add_dissonant_pitches_to_one_voice(
        self,
        voice_idx: int,
        voice: tuple,
        harmonic_frame_abstract: tuple,
        harmonic_frame_converted: tuple,
        harmonic_index_per_beat: tuple,
    ) -> tuple:
        # TODO(split functions in subfunctions, add better documentation and explanantion
        # whats actually happening)
        # TODO(add possibility to ignore harmonic borders in transitions and adding
        # tones anywhere between two consonant pitches)

        if self._add_dissonant_pitches_to_nth_voice[voice_idx]:

            pitches = []
            absolute_beats = []
            is_not_dissonant_pitch = []

            for pitch0, pitch1, absolute_position0, absolute_position1 in zip(
                voice[0],
                voice[0][1:],
                voice[1].convert2absolute(),
                voice[1].convert2absolute()[1:],
            ):

                absolute_position0 = int(absolute_position0)
                absolute_position1 = int(absolute_position1)

                pitches.append(pitch0)
                absolute_beats.append(absolute_position0)
                is_not_dissonant_pitch.append(True)

                if pitch0 is not mel.TheEmptyPitch and pitch1 is not mel.TheEmptyPitch:
                    distance = pitch1 - pitch0
                    distance_ct = distance.cents

                else:
                    distance_ct = 0

                if abs(distance_ct) > self._stepsize[1]:

                    if distance_ct > 0:
                        is_rising = True
                    else:
                        is_rising = False

                    available_harmony_indices_per_beat = harmonic_index_per_beat[
                        absolute_position0:absolute_position1
                    ]

                    area_per_harmony = collections.Counter(
                        available_harmony_indices_per_beat
                    )
                    area_per_harmony = tuple(
                        area_per_harmony[harmony_idx]
                        for harmony_idx in sorted(area_per_harmony)
                    )
                    area_per_harmony = tuple(
                        absolute_position0 + position
                        for position in tools.accumulate_from_zero(area_per_harmony)
                    )
                    area_per_harmony = tuple(
                        (a, b) for a, b in zip(area_per_harmony, area_per_harmony[1:])
                    )

                    available_harmony_indices = tuple(
                        sorted(set(available_harmony_indices_per_beat))
                    )
                    dissonant_pitches_of_available_harmonies = tuple(
                        self.__set_pitches_to_correct_frame(
                            harmonic_frame_converted[idx][1], pitch0, pitch1
                        )
                        for idx in available_harmony_indices
                    )
                    n_available_harmonies = len(available_harmony_indices)

                    found_pitches_per_harmony = [
                        [] for harmony in available_harmony_indices
                    ]

                    last_pitch = pitch0

                    nth_harmony = 0
                    while True:
                        filtered_pitches_per_harmony = tuple(
                            self.__filter_pitches_that_are_in_right_direction(
                                last_pitch,
                                dissonant_pitches_of_available_harmonies[nth_har],
                                is_rising,
                            )
                            for nth_har in range(nth_harmony, n_available_harmonies)
                        )

                        pitches_per_harmony_in_step_distance = tuple(
                            self.__filter_pitches_that_are_in_step_distance(
                                filtered_pitches
                            )
                            for filtered_pitches in filtered_pitches_per_harmony
                        )

                        if any(pitches_per_harmony_in_step_distance):
                            # try to use pitches that are within step distance
                            # towards the previous pitch.
                            for n_higher_harmony, pitches_in_step_distance in enumerate(
                                pitches_per_harmony_in_step_distance
                            ):
                                if pitches_in_step_distance:
                                    nth_harmony += n_higher_harmony
                                    break

                            choosen_pitch_index = tools.find_closest_index(
                                self._stepsize_centre,
                                tuple(
                                    pitch_and_cent_pair[1]
                                    for pitch_and_cent_pair in pitches_in_step_distance
                                ),
                            )
                            choosen_pitch = pitches_in_step_distance[
                                choosen_pitch_index
                            ][0]
                            found_pitches_per_harmony[nth_harmony].append(choosen_pitch)
                            last_pitch = choosen_pitch

                        elif any(filtered_pitches_per_harmony):
                            # in case there are no pitches within step distance,
                            # try to find the closest pitch outside of step
                            # distance
                            harmony_pitch_index_and_cent_pairs = []
                            for harmony_idx, filtered_pitches in enumerate(
                                filtered_pitches_per_harmony
                            ):
                                for pitch_idx, pitch_and_cent_pair in enumerate(
                                    filtered_pitches
                                ):
                                    harmony_pitch_index_and_cent_pairs.append(
                                        (
                                            (harmony_idx, pitch_idx),
                                            pitch_and_cent_pair[1],
                                        )
                                    )

                            choosen_pitch_data = sorted(
                                harmony_pitch_index_and_cent_pairs,
                                key=operator.itemgetter(1),
                            )[0]
                            nth_harmony += choosen_pitch_data[0][0]
                            choosen_pitch = filtered_pitches_per_harmony[
                                choosen_pitch_data[0][0]
                            ][choosen_pitch_data[0][1]][0]
                            found_pitches_per_harmony[nth_harmony].append(choosen_pitch)
                            last_pitch = choosen_pitch

                        else:
                            break

                    found_pitches = functools.reduce(
                        operator.add, found_pitches_per_harmony
                    )

                    # filter out pitches until there are as many pitches as beats
                    # if diff < 0:
                    # TODO(implement solution for this case)
                    # ACTUALLY THIS HAS TO BE DONE IN THE NEXT LOOP.
                    # EVEN IF GLOBALLY THERE IS ENOUGH SPACE / THERE ARE ENOUGH
                    # BEATS FOR ALL PITCHES, IT COULD STILL HAPPEN THAT WITHIN
                    # ONE HARMONY THERE ISN'T ENOUGH SPACE FOR ALL PITCHES
                    # APPEARING IN THIS HARMONY; WHILE ANOTHER HARMONY MAY BE
                    # COMPLETELY EMPTY.
                    # raise NotImplementedError

                    for found_pitches, area in zip(
                        found_pitches_per_harmony, area_per_harmony
                    ):

                        # find beats with the highest value in the respective area
                        weights_per_beat = self._weights_per_beat[slice(*area)]
                        divided_weights_per_beat = tuple(
                            tools.find_all_indices_of_n(weight, weights_per_beat)
                            for weight in sorted(set(weights_per_beat), reverse=True)
                        )

                        found_beats = []
                        beats2find = len(found_pitches) + 1
                        for division in divided_weights_per_beat:
                            if beats2find == 0:
                                break

                            division_size = len(division)
                            if division_size <= beats2find:
                                beats2find -= division_size
                                found_beats.extend(division)

                            else:
                                for idx in tools.accumulate_from_zero(
                                    tools.euclid(division_size, beats2find)
                                )[:-1]:
                                    found_beats.append(division[idx])

                                beats2find = 0

                        for pitch, beat in zip(found_pitches, found_beats[1:]):
                            pitches.append(pitch)
                            absolute_beats.append(beat + area[0])
                            is_not_dissonant_pitch.append(False)

                    # delete last dissonant pitch if it is too close to the next
                    # main pitch.
                    if len(found_pitches) > 0:
                        if abs((pitch1 - pitches[-1]).cents) < self._stepsize[-1]:
                            del pitches[-1]
                            del absolute_beats[-1]
                            del is_not_dissonant_pitch[-1]

            pitches.append(voice[0][-1])
            absolute_beats.append(voice[1].convert2absolute()[-1])
            is_not_dissonant_pitch.append(True)

            data = (
                tuple(pitches),
                binr.Compound(absolute_beats + [voice[1].beats]).convert2relative(),
                tuple(is_not_dissonant_pitch),
            )

            # change result by user definied constrains
            for constrain in self._constraints_added_pitches:
                data = constrain(voice_idx, data)

            return data

        else:

            return voice + (tuple(True for i in voice[0]),)

    def _add_dissonant_pitches(
        self,
        voices: tuple,
        harmonic_frame_abstract: tuple,
        harmonic_frame_converted: tuple,
        harmonic_index_per_beat: tuple,
    ) -> tuple:

        return tuple(
            self.__add_dissonant_pitches_to_one_voice(
                vox_idx,
                vox,
                harmonic_frame_abstract,
                harmonic_frame_converted,
                harmonic_index_per_beat,
            )
            for vox_idx, vox in enumerate(voices)
        )


class RhythmicCP(Counterpoint):
    """Class to model counterpoint that is based on a backtracking algorithm.

        MANDATORY:

        (1) a tuple containing harmonies, where every harmony is composed of:
            [a] BlueprintHarmony (the actual harmony)
            [b] voices to skip (voices that are silent)
            [c] BlueprintHarmony (optional dissonant pitches per harmony)
                => (HarmonyItself, Voices2skip, AdditionalDissonantPitches)
            - every harmony has to contain the same amount of pitches!

        (2) rhythm per voice
            - there has to be as many elements as there are voices
            - use a binr.Compound object for each input
            - every element has to have equal size (sum of durations)
            - depending on the particular subclass this either indicates the
              possible points where a new tone could be played or each rhythm is
              equal with the resulting rhythm of each voice


        OPTIONAL:

        (3) weight per beat
            - has to have equal size like the rhythms per voice
            - a list containing floating point numbers 1 >= x >= 0 (percentage)

        (4) constraints for harmonic resolution
            - functions for the harmonic frame algorithm
            - each function should have as an input:
                * counterpoint: the current counterpoint object
                * harmonies: abstract (tuple)
                * harmonies: concrete (tuple)
                * rhythmic weight (float)
                * duration of the current harmony (int)
                * harmony counter (int)
                * how many harmonies have to be found (int)

        (5) constraints for dissonant added pitches

        (6) step definition in cents. default: (70, 270)

        (7) ambitus_maker. default: make symmetrical ambitus with inner point 1/1,
            range 3/1 and overlap 5/4

        (8) start harmony: the first harmony (abstract form) -> if a solution can be
            found with this harmony
    """

    def __init__(
        self,
        harmonies: tuple,
        rhythm_per_voice: tuple,
        weights_per_beat: tuple = None,
        constraints_harmonic_resolution: tuple = tuple([]),
        constraints_added_pitches: tuple = tuple([]),
        stepsize: tuple = (70, 270),
        ambitus_maker: ambitus.AmbitusMaker = ambitus.SymmetricalRanges(
            ji.r(1, 1), ji.r(3, 1), ji.r(5, 4)
        ),
        start_harmony: tuple = None,
        use_sorting_algorithm: bool = True,
        add_dissonant_pitches_to_nth_voice: tuple = None,
    ) -> None:

        if not constraints_harmonic_resolution:
            constraints_harmonic_resolution = (
                constraints.HR_forbid_too_empty_harmonies(1),
            )

        self._constraints_harmonic_resolution = constraints_harmonic_resolution
        self._use_sorting_algorithm = use_sorting_algorithm

        super().__init__(
            harmonies,
            rhythm_per_voice,
            weights_per_beat,
            constraints_added_pitches,
            stepsize,
            ambitus_maker,
            start_harmony,
            add_dissonant_pitches_to_nth_voice,
        )

        self._n_harmonic_changes = len(self._attack_voice_pairs)

    @staticmethod
    def find_harmonic_index_per_beat(duration: int, harmonic_changes: tuple) -> tuple:
        """Identify the harmony for each beat.

        Return a tuple thats filled with as many integers as there are beats.
        Each integer is the index for the respective nth harmony.
        """
        ig0 = operator.itemgetter(0)
        positions = tuple(ig0(change) for change in harmonic_changes)
        return tuple(bisect.bisect_right(positions, index) for index in range(duration))

    #                                                                           ##
    # methods for the harmonic frame finding backtracking algorithm:            ##
    # (private since they won't be needed outside)                              ##
    #                                                                           ##

    def __make_harmonies_from_solution_and_indices(
        self, possible_solutions_per_item: tuple, indices_of_choosen_solutions: tuple
    ) -> tuple:
        return tuple(
            sol[idx]
            for idx, sol in zip(
                indices_of_choosen_solutions, possible_solutions_per_item
            )
        )

    def __is_valid(
        self,
        primes: tuple,
        possible_solutions_per_item: tuple,
        indices_of_choosen_solutions: tuple,
    ) -> bool:
        """Test if the choosen solutions are valid, taking in account constraints."""

        # check first if all solutions contain elements
        for solutions in possible_solutions_per_item:
            if not solutions:
                return False

        # generate necessary attributes for constrain calls
        abstract_harmonies = self.__make_harmonies_from_solution_and_indices(
            possible_solutions_per_item, indices_of_choosen_solutions
        )
        converted_harmonies = self.convert_abstract_harmonies(
            primes, abstract_harmonies
        )

        position = len(abstract_harmonies) - 1
        absolute_position = self._attack_voice_pairs[position][0]
        weight = self._weights_per_beat[absolute_position]

        try:
            duration = self._attack_voice_pairs[position + 1][0] - absolute_position
        except IndexError:
            duration = self._duration - absolute_position

        return all(
            constrain(
                self,
                abstract_harmonies,
                converted_harmonies,
                weight,
                duration,
                position,
                self._n_harmonic_changes,
            )
            for constrain in self._constraints_harmonic_resolution
        )

    def __sort_possible_solutions(
        self, primes: tuple, harmonies: tuple, former_choosen_harmonies: tuple
    ) -> tuple:
        """Sorts possible harmonies that could follow the former one.

        The harmonies are sorted in a way that generally prefered solutions
        may be tried first by the backtracking algorithm. Solutions are
        defined as generally preferable if they contain:

            (1) less empty voices...
            (2) less pitch repetitions between different voices in the same harmony...
            (3) less shared pitches with the former harmonies...

        than other harmonies.

        (where (1) is more important than (2) and (2) is more important than (3))

        In case the user may be searching for harmonic solutions with for instance
        many pitch repetitions, this sorting function may result in less efficent
        calculations. Therefore the sorting algorithm can be turned off during
        initalization.
        """

        def divide_by_n(harmonies: tuple, detect_attribute_in_harmonies) -> tuple:
            n_X_per_harmony = tuple(len(h[1]) for h in harmonies)
            sorted_unique_n_X = sorted(set(n_X_per_harmony))
            n_divisions = len(sorted_unique_n_X)
            divided_harmonies = [[] for n in range(n_divisions)]

            for harmony, X in zip(harmonies, n_X_per_harmony):
                divided_harmonies[sorted_unique_n_X.index(X)].append(harmony)

            return tuple(tuple(h) for h in divided_harmonies)

        def divide_by_n_empty_pitches(harmonies: tuple) -> tuple:
            def func(harmonies) -> tuple:
                return tuple(len(h[1]) for h in harmonies)

            return divide_by_n(harmonies, func)

        def divide_by_pitch_repetitions(harmonies: tuple) -> tuple:
            """Divide by vertial pitch repetitions. (the less the better)"""

            def func(harmonies) -> tuple:
                return tuple(harmony[0].n_pitch_repetitions for harmony in harmonies)

            return divide_by_n(harmonies, func)

        def divide_by_frequency(harmonies: tuple) -> tuple:
            """Divide by horizontal pitch repetitions. (the less the better)"""

            def func(harmonies) -> tuple:
                return tuple(
                    sum(
                        tuple(
                            harmony[0].n_common_pitches(former_harmony[0])
                            for former_harmony in former_choosen_harmonies
                        )
                    )
                    for harmony in harmonies
                )

            return divide_by_n(harmonies, func)

        def by_division_sorter(iterable: tuple, *functions) -> tuple:
            def recursive_sorter(iterable: tuple, remaining_functions: tuple) -> tuple:
                if remaining_functions:
                    return tuple(
                        recursive_sorter(sub, remaining_functions[1:])
                        for sub in remaining_functions[0](iterable)
                    )
                else:
                    return iterable

            sorted_iterable = recursive_sorter(iterable, functions)
            for i in range(len(functions)):
                sorted_iterable = functools.reduce(operator.add, sorted_iterable)

            return tuple(sorted_iterable)

        return by_division_sorter(
            harmonies,
            divide_by_n_empty_pitches,
            divide_by_pitch_repetitions,
            divide_by_frequency,
        )

    def __find_new_solutions(
        self,
        primes: tuple,
        possible_solutions_per_item: tuple,
        indices_of_choosen_solutions: tuple,
        position: int,
    ) -> tuple:
        """Return possible solutions for the next beat regarding former decisions."""

        former_harmonies = self.__make_harmonies_from_solution_and_indices(
            possible_solutions_per_item, indices_of_choosen_solutions
        )
        current_pitch_change = self._attack_voice_pairs[position][1]

        solutions_for_last_harmony = self._harmonic_network[former_harmonies[-1]]

        try:
            solutions_for_particular_pitch_change = tuple(
                solution[0]
                for solution in solutions_for_last_harmony[current_pitch_change]
            )
        except KeyError:
            solutions_for_particular_pitch_change = tuple([])

        if solutions_for_particular_pitch_change and self._use_sorting_algorithm:
            solutions_for_particular_pitch_change = self.__sort_possible_solutions(
                primes, solutions_for_particular_pitch_change, former_harmonies
            )

        return solutions_for_particular_pitch_change

    def _find_harmonic_frame(self, primes: tuple) -> tuple:
        """Return tuple that contains two subtuples.

        The first tuple contains one harmony for each change of pitch.
        The second tuple contains rhythms per voice.
        """

        possible_solutions_per_item = [
            self.__sort_possible_solutions(primes, self._harmonies, tuple([]))
        ]

        if self._start_harmony:
            possible_solutions_per_item[0] = list(possible_solutions_per_item[0])

            idx_start_harmony = possible_solutions_per_item[0].index(
                self._start_harmony
            )
            former_first_item = tuple(possible_solutions_per_item[0][0])

            possible_solutions_per_item[0][0] = self._start_harmony
            possible_solutions_per_item[0][idx_start_harmony] = former_first_item

            possible_solutions_per_item[0] = tuple(possible_solutions_per_item[0])

        indices_of_choosen_solutions = [0]

        while True:

            is_valid = self.__is_valid(
                primes, possible_solutions_per_item, indices_of_choosen_solutions
            )

            if is_valid:
                position = len(indices_of_choosen_solutions)
                if position < self._n_harmonic_changes:
                    possible_solutions_per_item.append(
                        self.__find_new_solutions(
                            primes,
                            possible_solutions_per_item,
                            indices_of_choosen_solutions,
                            position,
                        )
                    )
                    indices_of_choosen_solutions.append(0)
                else:
                    break

            else:

                while indices_of_choosen_solutions[-1] + 1 == len(
                    possible_solutions_per_item[-1]
                ):
                    indices_of_choosen_solutions = indices_of_choosen_solutions[:-1]
                    possible_solutions_per_item = possible_solutions_per_item[:-1]
                    if len(indices_of_choosen_solutions) == 0:
                        raise NotImplementedError("No solution could be found.")

                indices_of_choosen_solutions[-1] += 1

        harmonies = self.__make_harmonies_from_solution_and_indices(
            possible_solutions_per_item, indices_of_choosen_solutions
        )

        harmonic_index_per_beat = self.find_harmonic_index_per_beat(
            self._duration, self._attack_voice_pairs
        )

        return harmonies, self._possible_attacks_per_voice, harmonic_index_per_beat


class FreeStyleCP(Counterpoint):
    available_decision_types = ("activity", "random")

    def __init__(
        self,
        harmonies: tuple,
        possible_attacks_per_voice: tuple,
        weights_per_beat: tuple = None,
        constraints_added_pitches: tuple = tuple([]),
        stepsize: tuple = (70, 270),
        ambitus_maker: ambitus.AmbitusMaker = ambitus.SymmetricalRanges(
            ji.r(1, 1), ji.r(3, 1), ji.r(5, 4)
        ),
        start_harmony: tuple = None,
        use_sorting_algorithm: bool = True,
        add_dissonant_pitches_to_nth_voice: tuple = None,
        energy_per_voice: tuple = None,
        silence_decider_per_voice: tuple = None,
        weight_range: tuple = (1, 10),
        decision_type: str = "activity",
        random_seed: int = 100,
    ) -> None:
        super().__init__(
            harmonies,
            possible_attacks_per_voice,
            weights_per_beat,
            constraints_added_pitches,
            stepsize,
            ambitus_maker,
            start_harmony,
            add_dissonant_pitches_to_nth_voice,
        )

        try:
            assert decision_type in self.available_decision_types
        except AssertionError:
            msg = "Unknown decision_type '{}'. The only available types are '{}'.".format(
                decision_type, self.available_decision_types
            )
            raise NotImplementedError(msg)

        if energy_per_voice is None:
            energy_per_voice = tuple(10 for i in range(self._n_voices))

        if silence_decider_per_voice is None:
            silence_decider_per_voice = tuple(
                itertools.cycle((0,)) for i in range(self._n_voices)
            )

        self._silence_decider_per_voice = silence_decider_per_voice
        self._decision_type = decision_type

        if decision_type == self.available_decision_types[0]:
            self._add_activity_al = activity_levels.ActivityLevel()
            self._energy_lv_per_beat = self.convert_weights_per_beat2activity_lv_per_beat(
                self._weights_per_beat, weight_range
            )
            self._al_per_voice = tuple(
                infit.ActivityLevel(lv) for lv in energy_per_voice
            )

        elif decision_type == self.available_decision_types[1]:
            import random

            random.seed(random_seed)
            self._random_unit = random
            self._border_per_voice = tuple(lv / 10 for lv in energy_per_voice)
            self._weights_per_beat = tools.scale(self._weights_per_beat, *weight_range)

    @staticmethod
    def convert_weights_per_beat2activity_lv_per_beat(
        weights_per_beat: tuple, range=(1, 10)
    ) -> tuple:
        different_weights = tuple(sorted(set(weights_per_beat)))
        n_different_weights = len(different_weights)
        level_per_weight = tools.accumulate_from_n(
            tools.euclid(range[1] - range[0], n_different_weights), range[0]
        )
        return tuple(
            level_per_weight[different_weights.index(weight)]
            for weight in weights_per_beat
        )

    @staticmethod
    def __detect_if_voices_could_become_silent(
        possible_solutions: tuple, n_voices: int
    ) -> tuple:
        """Return tuple that contains True value if voice could become silence"""
        voice_could_become_silent = [False for i in range(n_voices)]
        for solution in possible_solutions:
            for v_idx, is_next_tone_a_pitch in enumerate(solution[1]):
                if not is_next_tone_a_pitch:
                    voice_could_become_silent[v_idx] = True
        return tuple(voice_could_become_silent)

    @staticmethod
    def __sort_solutions_by_least_often_used_pitches(
        possible_solutions: tuple, used_pitches_counter: collections.Counter
    ) -> tuple:
        solution_point_pairs = []
        for solution in possible_solutions:
            points = sum(used_pitches_counter[pitch] for pitch in solution[0].blueprint)
            points += len(solution[1]) * used_pitches_counter[mel.TheEmptyPitch]
            solution_point_pairs.append((solution, points))

        sorted_solutions = tuple(
            pair[0] for pair in sorted(solution_point_pairs, key=operator.itemgetter(1))
        )

        return sorted_solutions

    def __choose_harmony(
        self,
        possible_solutions: tuple,
        voices_would_become_tone: tuple,
        used_pitches_counter: dict,
    ) -> tuple:
        n_voices = len(voices_would_become_tone)

        # use those solutions that are most similar to the expected
        # result (that indicates which voices shall turn silent and which
        # voices shall get a new pitch in the next harmony)
        difference_per_solution = []
        for solution in possible_solutions:
            difference = sum(
                0 if a == b else 1
                for a, b in zip(voices_would_become_tone, solution[1])
            )
            difference_per_solution.append(difference)

        min_difference = min(difference_per_solution)

        if min_difference <= n_voices - 1:

            choosen_solutions = tuple(
                solution
                for difference, solution in zip(
                    difference_per_solution, possible_solutions
                )
                if difference == min_difference
            )

            # only use the harmonies and throw away the tuple that
            # indicates if in the new harmony the changed voices will
            # be either silent or get a new pitch
            choosen_solutions = tuple(solution[0] for solution in choosen_solutions)

            solutions = self.__sort_solutions_by_least_often_used_pitches(
                choosen_solutions, used_pitches_counter
            )
            return solutions[0]

        return tuple([])

    def _find_harmonic_frame(self, primes: tuple) -> tuple:
        """Return tuple that contains two subtuples.

        The first tuple contains one harmony for each change of pitch.
        The second tuple contains rhythms per voice.
        """

        # TODO(add better documentation what is actually happening)

        if self._start_harmony:
            harmonies = [self._start_harmony]

        else:
            harmonies = [self.harmonies[0]]

        attack_positions_per_voice = [[0] for i in range(self._n_voices)]

        attack_voice_pairs = self._attack_voice_pairs[1:]

        used_pitches_counter = collections.Counter(
            harmonies[0][0].blueprint
            + tuple(mel.TheEmptyPitch for i in harmonies[0][1])
        )

        harmonic_index_per_beat = []

        while True:
            if not attack_voice_pairs:
                break

            position, voices = attack_voice_pairs[0]
            x_voices = tuple(range(len(voices)))
            possible_combinations = functools.reduce(
                operator.add,
                tuple(tuple(itertools.combinations(voices, n + 1)) for n in x_voices),
            )
            solutions_per_combination = tuple(
                self._harmonic_network[harmonies[-1]][combination]
                for combination in possible_combinations
            )

            if any(solutions_per_combination):

                if self._decision_type == self.available_decision_types[0]:
                    activity_test = self._add_activity_al(
                        self._energy_lv_per_beat[position]
                    )

                elif self._decision_type == self.available_decision_types[1]:
                    activity_test = (
                        self._random_unit.random() < self._weights_per_beat[position]
                    )

                if activity_test:

                    if self._decision_type == self.available_decision_types[0]:
                        nth_voice_shall_be_activated = tuple(
                            next(self._al_per_voice[vox_idx]) for vox_idx in voices
                        )

                    elif self._decision_type == self.available_decision_types[1]:
                        nth_voice_shall_be_activated = tuple(
                            self._random_unit.random() < self._border_per_voice[vox_idx]
                            for vox_idx in voices
                        )

                    if any(nth_voice_shall_be_activated):
                        used_voices = tuple(
                            voices[idx]
                            for idx, shall_activate in enumerate(
                                nth_voice_shall_be_activated
                            )
                            if shall_activate
                        )
                        n_voices = len(used_voices)
                        possible_solutions = solutions_per_combination[
                            possible_combinations.index(used_voices)
                        ]

                        could_become_silent = self.__detect_if_voices_could_become_silent(
                            possible_solutions, n_voices
                        )

                        voice_would_become_silent = tuple(
                            next(self._silence_decider_per_voice[vox_idx])
                            if could_become
                            else False
                            for vox_idx, could_become in zip(
                                used_voices, could_become_silent
                            )
                        )

                        voices_would_become_tone = tuple(
                            not boolean for boolean in voice_would_become_silent
                        )

                        choosen_harmony = self.__choose_harmony(
                            possible_solutions,
                            voices_would_become_tone,
                            used_pitches_counter,
                        )

                        if choosen_harmony:
                            harmonies.append(choosen_harmony)
                            for vox_idx in used_voices:
                                attack_positions_per_voice[vox_idx].append(position)

                            harmonic_index_per_beat.append(position)

                            for pitch in choosen_harmony[0].blueprint:
                                used_pitches_counter.update({pitch: 1})

                            used_pitches_counter.update(
                                {mel.TheEmptyPitch: len(choosen_harmony[1])}
                            )

            attack_voice_pairs = attack_voice_pairs[1:]

        rhythm_per_voice = tuple(
            binr.Compound(
                tuple((b - a for a, b in zip(attacks, attacks[1:] + [self._duration])))
            )
            for attacks in attack_positions_per_voice
        )

        harmonic_index_per_beat.append(self._duration)

        if len(harmonic_index_per_beat) > 1:
            harmonic_index_per_beat = tuple(
                b - a
                for a, b in zip(harmonic_index_per_beat, harmonic_index_per_beat[1:])
            )

        harmonic_index_per_beat = functools.reduce(
            operator.add,
            tuple(
                tuple(idx for i in range(duration))
                for idx, duration in enumerate(harmonic_index_per_beat)
            ),
        )

        return tuple(harmonies), rhythm_per_voice, harmonic_index_per_beat
