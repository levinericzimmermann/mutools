import bisect
import collections
import functools
import itertools
import operator

from mu.mel import ji
from mu.mel import mel

from mu.rhy import binr

from mu.utils import tools

from mutools import ambitus


"""This module contains different algorithms that model counterpoint-like structures."""


class RhythmicCP(object):
    """Class to model Counterpoint that is based on rhythmic material.

    Arguments for RhythmicCP:

        MANDATORY:
        (1) a tuple containing harmonies, where every harmony is composed of:
            [a] BlueprintHarmony (the actual harmony)
            [b] voices to skip (voices that are silent)
            [c] BlueprintHarmony (optional dissonant pitches per harmony)
                => (HarmonyItself, Voices2skip, AdditionalDissonantPitches)
            - every harmony has to contain the same amount of pitches!
        (2) rhythms per voice
            - there has to be as many elements as there are voices
            - use a binr.Compound object for each input
            - every element has to have equal size (sum of durations)

        OPTIONAL:
        (3) weight per beat
            - has to have equal size like the rhythms per voice
            - a list containing floating point numbers 1 >= x >= 0 (percentage)
        (4) constraints for harmonic resolution
            - functions for the harmonic frame algorithm
            - each function should have as an input:
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

    FUNCTION CALL WITH PRIME NUMBERS TO RESOLVE ABSTRACT HARMONIES
        (2) convert each voice to a list of (pitches, binr.Compound)
            - find best voice leading for each voice regarding the ambitus
            - if there are rests in the voice, start the best voice leading algorithm
              again after every rest
        (3) add additional dissonant pitches
            - check following pitches in one voice: if they are bigger than a step
              continue
            - identify the used harmonies between the two pitches
            - find their dissonant pitches and normalize them to the respective octaves
            - filter only those pitches that are inbetween the following pitches
            - if there are pitches available
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
    ) -> None:

        # testing for correct arguments
        self.test_arguments_for_equal_size(
            "Harmony", harmonies, lambda harmony: sum(len(h) for h in harmony[:2])
        )
        self.test_arguments_for_equal_size(
            "Rhythm", rhythm_per_voice, lambda rhythm: sum(rhythm)
        )

        self.__duration = rhythm_per_voice[0].beats
        self.__n_voices = len(rhythm_per_voice)

        if weights_per_beat is None:
            weights_per_beat = tuple(1 for i in range(self.__duration))

        # TODO(Check later which of those attributes are really used!)
        self.__start_harmony = start_harmony
        self.__stepsize = stepsize
        self.__stepsize_centre = stepsize[1] - stepsize[0]
        self.__ambitus_per_voice = ambitus_maker(self.__n_voices)
        self.__weights_per_beat = weights_per_beat
        self.__constraints_harmonic_resolution = constraints_harmonic_resolution
        self.__constraints_added_pitches = constraints_added_pitches
        self.__harmonies = harmonies
        self.__rhythm_per_voice = rhythm_per_voice
        self.__use_sorting_algorithm = use_sorting_algorithm

        self.__harmonic_network = self.make_harmonic_network(
            self.__n_voices, harmonies, harmonies
        )
        self.__harmonic_changes = self.find_harmonic_changes(rhythm_per_voice)
        self.__harmonic_index_per_beat = self.find_harmonic_index_per_beat(
            self.__duration, self.__harmonic_changes
        )
        self.__n_harmonic_changes = len(self.__harmonic_changes)

    @staticmethod
    def find_harmonic_index_per_beat(duration: int, harmonic_changes: tuple) -> tuple:
        """Identify the harmony for each beat.

        Return a tuple thats filled with as many integers as there are beats.
        Each integer is the index for the respective nth harmony.
        """
        ig0 = operator.itemgetter(0)
        positions = tuple(ig0(change) for change in harmonic_changes)
        return tuple(bisect.bisect_right(positions, index) for index in range(duration))

    @staticmethod
    def find_harmonic_changes(rhythm_per_voice: tuple) -> tuple:
        """Find which voice is changing its pitch at which position with a new harmony.

        Return tuple filled with tuple. The first element of those subtuples is the
        absolute position of the harmonic change and the second element contains the
        indices of all voices whose pitch is changing with this harmonic change.
        """
        harmonic_changes = []
        for position in functools.reduce(
            lambda r0, r1: r0.union(r1), rhythm_per_voice
        ).convert2absolute():
            relevant_voices = []
            for vox_idx, voice in enumerate(rhythm_per_voice):
                if position in voice.convert2absolute():
                    relevant_voices.append(vox_idx)

            assert int(position) == position
            harmonic_changes.append((int(position), tuple(relevant_voices)))
        return tuple(harmonic_changes)

    @staticmethod
    def make_harmonic_network(n_voices: int, departure: tuple, arrival: tuple) -> dict:
        """Return dict where every harmony has one entry.

        Each entry is linked to a dict where the keys
        are tuples that indicate the voices that will
        change and the values are tuples that
        contain all possible following harmonies for this
        particular change.

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
                    possible_arrivals[changed_pitches].append(arriving_harmony)

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

    def __extract_harmonic_frame_voices(self, converted_harmonic_frame: tuple) -> tuple:
        ig0 = operator.itemgetter(0)
        only_harmonies = tuple(
            ig0(harmony_and_additional_dissonant_pitches)
            for harmony_and_additional_dissonant_pitches in converted_harmonic_frame
        )

        pitches_per_voice = []
        for amb, pitches in zip(self.__ambitus_per_voice, tuple(zip(*only_harmonies))):
            pitches_without_repetitions = [pitches[0]]
            for pitch in pitches[1:]:
                if pitches_without_repetitions[-1] != pitch:
                    pitches_without_repetitions.append(pitch)
            pitches_per_voice.append(
                amb.find_best_voice_leading(pitches_without_repetitions)
            )

        return tuple(
            (pitches, rhythm)
            for pitches, rhythm in zip(pitches_per_voice, self.__rhythm_per_voice)
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
        harmonic_frame_abstract = self.__find_harmonic_frame(primes)
        harmonic_frame_converted = self.convert_abstract_harmonies(
            primes, harmonic_frame_abstract
        )

        # (2) extract harmonic frame voices, put pitches in correct octave
        voices_without_added_pitches = self.__extract_harmonic_frame_voices(
            harmonic_frame_converted
        )

        # (3) add additional dissonant pitches (interpolation)
        voices_with_added_pitches = self.__add_dissonant_pitches(
            voices_without_added_pitches,
            harmonic_frame_abstract,
            harmonic_frame_converted,
        )

        return voices_with_added_pitches, voices_without_added_pitches

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
        absolute_position = self.__harmonic_changes[position][0]
        weight = self.__weights_per_beat[absolute_position]

        try:
            duration = self.__harmonic_changes[position + 1][0] - absolute_position
        except IndexError:
            duration = self.__duration - absolute_position

        return all(
            constrain(
                abstract_harmonies,
                converted_harmonies,
                weight,
                duration,
                position,
                self.__n_harmonic_changes,
            )
            for constrain in self.__constraints_harmonic_resolution
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
        current_pitch_change = self.__harmonic_changes[position][1]

        solutions_for_last_harmony = self.__harmonic_network[former_harmonies[-1]]

        try:
            solutions_for_particular_pitch_change = solutions_for_last_harmony[
                current_pitch_change
            ]
        except KeyError:
            solutions_for_particular_pitch_change = tuple([])

        if solutions_for_particular_pitch_change and self.__use_sorting_algorithm:
            solutions_for_particular_pitch_change = self.__sort_possible_solutions(
                primes, solutions_for_particular_pitch_change, former_harmonies
            )

        return solutions_for_particular_pitch_change

    def __find_harmonic_frame(self, primes: tuple) -> tuple:
        """Return tuple that contains one harmony for each change of pitch."""

        possible_solutions_per_item = [
            self.__sort_possible_solutions(primes, self.__harmonies, tuple([]))
        ]

        if self.__start_harmony:
            possible_solutions_per_item[0] = list(possible_solutions_per_item[0])

            idx_start_harmony = possible_solutions_per_item[0].index(
                self.__start_harmony
            )
            former_first_item = tuple(possible_solutions_per_item[0][0])

            possible_solutions_per_item[0][0] = self.__start_harmony
            possible_solutions_per_item[0][idx_start_harmony] = former_first_item

            possible_solutions_per_item[0] = tuple(possible_solutions_per_item[0])

        indices_of_choosen_solutions = [0]

        while True:

            is_valid = self.__is_valid(
                primes, possible_solutions_per_item, indices_of_choosen_solutions
            )

            if is_valid:
                position = len(indices_of_choosen_solutions)
                if position < self.__n_harmonic_changes:
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

        return self.__make_harmonies_from_solution_and_indices(
            possible_solutions_per_item, indices_of_choosen_solutions
        )

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
            if abs_ct > self.__stepsize[0] and abs_ct < self.__stepsize[1]:
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
        voice: tuple,
        harmonic_frame_abstract: tuple,
        harmonic_frame_converted: tuple,
    ) -> tuple:

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

            if abs(distance_ct) > self.__stepsize[1]:

                if distance_ct > 0:
                    is_rising = True
                else:
                    is_rising = False

                available_harmony_indices_per_beat = self.__harmonic_index_per_beat[
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
                            self.__stepsize_centre,
                            tuple(
                                pitch_and_cent_pair[1]
                                for pitch_and_cent_pair in pitches_in_step_distance
                            ),
                        )
                        choosen_pitch = pitches_in_step_distance[choosen_pitch_index][0]
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
                                    ((harmony_idx, pitch_idx), pitch_and_cent_pair[1])
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
                    weights_per_beat = self.__weights_per_beat[slice(*area)]
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
                    if abs((pitch1 - pitches[-1]).cents) < self.__stepsize[-1]:
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
        for constrain in self.__constraints_added_pitches:
            data = constrain(data)

        return data

    def __add_dissonant_pitches(
        self,
        voices: tuple,
        harmonic_frame_abstract: tuple,
        harmonic_frame_converted: tuple,
    ) -> tuple:

        return tuple(
            self.__add_dissonant_pitches_to_one_voice(
                vox, harmonic_frame_abstract, harmonic_frame_converted
            )
            for vox in voices
        )
