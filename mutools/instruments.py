import abc
import abjad
import bisect
import crosstrainer
import functools
import itertools
import operator

from mu.mel import edo
from mu.mel import ji
from mu.mel import mel


try:
    import quicktions as fractions
except ImportError:
    import fractions


class Instrument(abc.ABC):
    @abc.abstractmethod
    def convert2abjad_voices(self, cadence, grid) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def convert2abjad_pitches(self, pitches) -> list:
        raise NotImplementedError

    @staticmethod
    def mk_bar_line() -> abjad.LilyPondCommand:
        return abjad.LilyPondCommand('bar "|"', "after")

    @staticmethod
    def seperate_by_grid(delay, start, stop, absolute_grid, grid, grid_object) -> tuple:
        grid_start = bisect.bisect_right(absolute_grid, start) - 1
        grid_stop = bisect.bisect_right(absolute_grid, stop)
        passed_groups = tuple(range(grid_start, grid_stop, 1))

        # are_both_on_line = all((start in absolute_grid, stop in absolute_grid))

        if len(passed_groups) == 1:
            return (delay,)

        else:
            delays = []
            is_connectable_per_delay = []
            for i, group in enumerate(passed_groups):
                if i == 0:
                    diff = start - absolute_grid[group]
                    new_delay = grid[group] - diff
                    is_connectable = any(
                        (start in absolute_grid, new_delay.denominator <= 4)
                    )
                elif i == len(passed_groups) - 1:
                    new_delay = stop - absolute_grid[group]
                    is_connectable = any(
                        (stop in absolute_grid, new_delay.denominator <= 4)
                    )
                else:
                    new_delay = grid[group]
                    is_connectable = True

                if new_delay > 0:
                    delays.append(new_delay)
                    is_connectable_per_delay.append(is_connectable)

            ldelays = len(delays)
            if ldelays == 1:
                return tuple(delays)

            connectable_range = [int(not is_connectable_per_delay[0]), ldelays]
            if not is_connectable_per_delay[-1]:
                connectable_range[-1] -= 1

            solutions = [((n,), m) for n, m in enumerate(delays)]
            for item0 in range(*connectable_range):
                for item1 in range(item0 + 1, connectable_range[-1] + 1):
                    connected = sum(delays[item0:item1])
                    if abjad.Duration(connected).is_assignable:
                        sol = (tuple(range(item0, item1)), connected)
                        if sol not in solutions:
                            solutions.append(sol)

            possibilites = crosstrainer.Stack(fitness="min")
            amount_connectable_items = len(delays)
            has_found = False
            lsolrange = tuple(range(len(solutions)))
            for combsize in range(1, amount_connectable_items + 1):
                for comb in itertools.combinations(lsolrange, combsize):
                    pos = tuple(solutions[idx] for idx in comb)
                    items = functools.reduce(operator.add, tuple(p[0] for p in pos))
                    litems = len(items)
                    is_unique = litems == len(set(items))
                    if is_unique and litems == amount_connectable_items:
                        sorted_pos = tuple(
                            item[1] for item in sorted(pos, key=lambda x: x[0][0])
                        )
                        fitness = len(sorted_pos)
                        possibilites.append(sorted_pos, fitness)
                        has_found = True
                if has_found:
                    break

            result = possibilites.best[0]
            return tuple(result)

    @staticmethod
    def seperate_by_assignablity(duration, grid) -> tuple:
        def find_sum_in_numbers(numbers, solution, allowed_combinations) -> tuple:
            return tuple(
                sorted(c, reverse=True)
                for c in itertools.combinations_with_replacement(
                    numbers, allowed_combinations
                )
                if sum(c) == solution
            )

        if abjad.Duration(duration).is_assignable is True:
            return (abjad.Duration(duration),)

        numerator = duration.numerator
        denominator = duration.denominator
        possible_durations = [
            i
            for i in range(1, numerator + 1)
            if abjad.Duration(i, denominator).is_assignable is True
        ]
        solution = []
        c = 1
        while len(solution) == 0:
            if c == len(possible_durations):
                print(numerator, denominator)
                raise ValueError("Can't find any possible combination")
            s = find_sum_in_numbers(possible_durations, numerator, c)
            if s:
                solution.extend(s)
            else:
                c += 1
        hof = crosstrainer.Stack(size=1, fitness="max")
        for s in solution:
            avg_diff = sum(abs(a - b) for a, b in itertools.combinations(s, 2))
            hof.append(s, avg_diff)
        best = hof.best[0]
        return tuple(fractions.Fraction(s, denominator) for s in best)

    @staticmethod
    def apply_beams(notes, durations, absolute_grid) -> tuple:
        duration_positions = []
        absolute_durations = tuple(itertools.accumulate([0] + list(durations)))
        for dur in absolute_durations:
            pos = bisect.bisect_right(absolute_grid, dur) - 1
            duration_positions.append(pos)
        beam_indices = []
        current = None
        for idx, pos in enumerate(duration_positions):
            if pos != current:
                beam_indices.append(idx)
                current = pos
        for idx0, idx1 in zip(beam_indices, beam_indices[1:]):
            if idx1 == beam_indices[-1]:
                subnotes = notes[idx0:]
            else:
                subnotes = notes[idx0:idx1]
            add_beams = False
            for n in subnotes:
                if type(n) != abjad.Rest:
                    add_beams = True
                    break
            if len(subnotes) < 2:
                add_beams = False
            if add_beams is True:
                abjad.attach(abjad.Beam(), subnotes)
        return notes

    @staticmethod
    def mk_voice(voc):
        # add bar line
        # abjad.attach(Instrument.mk_bar_line(), voc[-1])
        return voc

    def convert_abjad_pitches_and_mu_rhythms2abjad_notes(
        self, harmonies: list, delays: list, grid
    ) -> list:
        leading_pulses = grid.leading_pulses
        absolute_leading_pulses = tuple(
            itertools.accumulate([0] + list(leading_pulses))
        )
        converted_delays = grid.apply_delay(delays)
        absolute_delays = tuple(itertools.accumulate([0] + list(converted_delays)))
        # 1. generate notes
        notes = abjad.Measure(abjad.TimeSignature(grid.absolute_meter), [])
        resulting_durations = []
        for harmony, delay, start, end in zip(
            harmonies, converted_delays, absolute_delays, absolute_delays[1:]
        ):
            subnotes = abjad.Voice()
            seperated_by_grid = Instrument.seperate_by_grid(
                delay, start, end, absolute_leading_pulses, leading_pulses, grid
            )
            assert sum(seperated_by_grid) == delay
            for d in seperated_by_grid:
                seperated_by_assignable = Instrument.seperate_by_assignablity(d, grid)
                assert sum(seperated_by_assignable) == d
                for assignable in seperated_by_assignable:
                    resulting_durations.append(assignable)
                    if harmony:
                        chord = abjad.Chord(harmony, abjad.Duration(assignable))
                    else:
                        chord = abjad.Rest(abjad.Duration(assignable))
                    subnotes.append(chord)
            if len(subnotes) > 1 and len(harmony) > 0:
                abjad.attach(abjad.Tie(), subnotes[:])
            notes.extend(subnotes)
        assert sum(resulting_durations) == sum(converted_delays)
        voice = Instrument.mk_voice(notes)
        # 2. apply beams
        voice = Instrument.apply_beams(
            voice, resulting_durations, absolute_leading_pulses
        )
        return voice

    @staticmethod
    def mk_no_time_signature() -> abjad.LilyPondCommand:
        return abjad.LilyPondCommand(
            "override Score.TimeSignature.stencil = ##f", "before"
        )

    @staticmethod
    def mk_numeric_ts() -> abjad.LilyPondCommand:
        return abjad.LilyPondCommand(
            "numericTimeSignature", "before"
        )

    @staticmethod
    def mk_cadenza() -> abjad.LilyPondCommand:
        return abjad.LilyPondCommand("cadenzaOn", "before")

    @staticmethod
    def mk_staff(voices, clef="percussion") -> abjad.Staff:
        staff = abjad.Staff([])
        for v in voices:
            staff.append(v)
        clef = abjad.Clef(clef)
        abjad.attach(clef, staff)
        abjad.attach(Instrument.mk_numeric_ts(), staff[0][0])
        # abjad.attach(Instrument.mk_no_time_signature(), staff[0])
        # abjad.attach(Instrument.mk_cadenza(), staff[0])
        return staff


class MonochordFret(object):
    def __init__(self, number, octave) -> None:
        self.number = number
        # self.octave = number // 120
        self.octave = octave

    def __hash__(self):
        return hash((self.number, self.octave))

    def __eq__(self, other):
        try:
            return all((self.number == other.number, self.octave == other.octave))
        except AttributeError:
            return False

    def copy(self):
        return type(self)(int(self.number), int(self.octave))

    def __repr__(self) -> str:
        return "Fret: {0}_{1}".format(self.number, self.octave)

    def convert2absolute_fret(self, divisions: int) -> int:
        offset = self.octave * divisions
        return self.number + offset

    def convert2relative_fret(self) -> str:
        if self.octave == 0:
            oc = "I"
        elif self.octave == 1:
            oc = "II"
        elif self.octave == 2:
            oc = "III"
        elif self.octave == 3:
            oc = "IV"
        else:
            raise NotImplementedError()

        return "{0} ({1})".format(self.number, oc)


class MonochordString(object):
    def __init__(self, number: int, pitch: ji.JIPitch, divisions=120):
        self.__number = number
        self.pitch = pitch
        self.divisions = divisions
        self.__division_class = edo.EdoPitch.mk_new_edo_class(2, divisions)
        """
        division_floats = [self.__division_class(i) for i in range(divisions)]
        for obj in division_floats:
            obj._concert_pitch = 1
        self.division_floats = [obj.freq for obj in division_floats]
        """
        self.division_floats = MonochordString.mk_division_floats()
        self.fret = self.find_best_fret_for_pitch(pitch)

    def copy(self):
        return type(self)(int(self.number), self.pitch.copy(), self.divisions)

    @staticmethod
    def mk_cents() -> tuple:
        cents = [
            0,
            6,
            15.3,
            23.8,
            32.5,
            42.9,
            50,
            60.7,
            70.5,
            78.5,
            87.5,
            98,
            107,
            116.25,
            126,
            136.5,
            147,
            154,
            164,
            171.5,
            182,
            195,
            205,
            212,
            221.5,
            231,
            241.8,
            252,
            261.5,
            270,
            281,
            290,
            300,
            310,
            319.5,
        ]
        while len(cents) < 120 * 3:
            cents.append(cents[-1] + 10)
        return tuple(cents)

    @staticmethod
    def mk_division_floats() -> tuple:
        cents = MonochordString.mk_cents()
        return tuple(mel.SimplePitch(1, ct).freq for ct in cents)

    @property
    def number(self):
        return self.__number

    def __repr__(self):
        return repr((self.number, self.fret, self.pitch))

    def __eq__(self, other):
        try:
            return all(
                (
                    self.number == other.number,
                    self.pitch == other.pitch,
                    self.fret == other.fret,
                )
            )
        except AttributeError:
            return False

    def find_best_fret_for_pitch(self, pitch):
        # octave = 0
        # comp = ji.r(2, 1)
        # while pitch.float >= comp.float:
        #     pitch -= comp
        #     octave += 1
        factor = pitch.float
        closest = bisect.bisect_right(self.division_floats, factor)
        possible_solutions = []
        for c in [closest - 1, closest, closest + 1]:
            try:
                possible_solutions.append(self.division_floats[c])
            except IndexError:
                pass
        hof = crosstrainer.MultiDimensionalRating(size=1, fitness=[-1])
        for pos in possible_solutions:
            if pos > factor:
                diff = pos / factor
            else:
                diff = factor / pos
            hof.append(pos, diff)
        number = self.division_floats.index(hof._items[0])
        return MonochordFret(number % 120, number // 120)


class Monochord(Instrument):
    transposition_percussion_clef = 10
    divisions = 120
    octaves = 3
    length_fretboard_first_octave = 60
    length_bridge = 1
    minimum_bridge_distance = 2
    minimum_distance_from_right_bridge = 9

    def __init__(self, strings: list) -> None:
        self.fret_distances = Monochord.fret_distances
        self.absolute_fret_distances = tuple(
            itertools.accumulate([0] + list(self.fret_distances))
        )
        self.strings = strings

    def copy(self):
        return type(self)([s.copy() for s in self.strings])

    def is_enough_space_for_frets(self, strings) -> bool:
        frets = tuple(s.fret for s in strings)
        frets_abs = tuple(fret.convert2absolute_fret(self.divisions) for fret in frets)
        fret_distances = [
            sum(self.fret_distances[f0:f1]) for f0, f1 in zip(frets_abs, frets_abs[1:])
        ]
        return all(d >= self.minimum_bridge_distance for d in fret_distances)

    @staticmethod
    def is_enough_space_for_pitch(pitch) -> bool:
        string = MonochordString(0, pitch, 120)
        fret_abs = string.fret.convert2absolute_fret(120)
        fret_distances = Monochord.fret_distances
        absolute_fret_distances = tuple(
            itertools.accumulate([0] + list(fret_distances))
        )
        real_distance = absolute_fret_distances[fret_abs]
        min_distance = Monochord.minimum_distance_from_right_bridge

        return real_distance >= min_distance

    def are_frets_far_enough_from_right_bridge(self, strings) -> bool:
        frets = tuple(s.fret for s in strings)
        frets_abs = tuple(fret.convert2absolute_fret(self.divisions) for fret in frets)
        fd = [self.absolute_fret_distances[fa] for fa in frets_abs]
        return all(d >= self.minimum_distance_from_right_bridge or d == 0 for d in fd)

    def convert2abjad_voices(self, cadence, grid, previous_strings=None) -> list:
        def tie_pauses(harmonies, delays):
            newh = []
            newd = []
            first = True
            for harmony, delay in zip(harmonies, delays):
                if not harmony and first is False:
                    newd[-1] += delay
                else:
                    newh.append(harmony)
                    newd.append(delay)
                first = False
            return newh, newd

        cadence = cadence.tie_pauses()
        pitches = self.convert2abjad_pitches(cadence.pitch)
        distributed_pitches = self.distribute_pitches_on_different_staves(pitches)
        distributed_pitches = [
            tie_pauses(distrp, cadence.delay) for distrp in distributed_pitches
        ]
        voices = [
            self.convert_abjad_pitches_and_mu_rhythms2abjad_notes(
                distr_p[0], distr_p[1], grid
            )
            for distr_p in distributed_pitches
        ]
        if previous_strings is not None:
            self.attach_fret_changes(voices, previous_strings)

        return voices

    def attach_fret_changes(self, voices, previous_strings):
        marks = []
        for s0, s1 in zip(self.strings, previous_strings):
            if s0.fret.convert2absolute_fret(120) != s1.fret.convert2absolute_fret(120):
                if s0.number != 20:
                    mark = Monochord.mk_fret_change_mark(
                        s0.number,
                        s0.fret.convert2relative_fret(),
                        s1.fret.convert2absolute_fret(120),
                        s0.fret.convert2absolute_fret(120),
                    )
                    marks.append(mark)
        if marks:
            composed_mark = Monochord.mk_fret_change_markup(marks)
            abjad.attach(composed_mark, voices[0][0])

    @property
    def strings(self):
        return self.__strings

    @strings.setter
    def strings(self, strings):
        def mk_tests(strings):
            # no double string
            string_numbers = tuple(s.number for s in strings)
            assert len(set(string_numbers)) == len(string_numbers)
            # frets are ascending
            frets = tuple(s.fret for s in strings)
            frets_abs = list(
                fret.convert2absolute_fret(self.divisions) for fret in frets
            )
            assert frets_abs == sorted(frets_abs)
            # no frets are too close
            # assert self.is_enough_space_for_frets(strings) is True
            # no frets too close to the right bridge (they don't work
            # anymore, if they are too close)
            # assert self.are_frets_far_enough_from_right_bridge(strings) is True

        mk_tests(strings)
        self.__strings = list(strings)

    @staticmethod
    def calculate_fret_lengths(divisions, length_fretboard) -> tuple:
        interval = pow(2, 1 / divisions)
        dlf = length_fretboard * 2
        absolute_distances = [
            dlf - (dlf * (1 / (interval ** i))) for i in range(divisions)
        ]
        return tuple(absolute_distances)

    @classmethod
    def calculate_fret_distances(cls) -> tuple:
        length_fretboards = [
            cls.length_fretboard_first_octave / (2 ** i) for i in range(cls.octaves)
        ]
        fret_lengths = [
            cls.calculate_fret_lengths(cls.divisions, l) for l in length_fretboards
        ]
        length_fretboards_acc = itertools.accumulate([0] + length_fretboards)
        fret_lengths = [
            tuple(map(lambda length: length + offset, lengths))
            for lengths, offset in zip(fret_lengths, length_fretboards_acc)
        ]
        fret_lenghts = functools.reduce(operator.add, fret_lengths)
        fret_distances = [b - a for a, b in zip(fret_lenghts, fret_lenghts[1:])]
        return fret_distances

    @staticmethod
    def distribute_pitches_on_different_staves(harmonies) -> tuple:
        border = -9
        voices = [[] for i in range(2)]
        for harmony in harmonies:
            new_chords = [[], []]
            for pitch in harmony:
                number = pitch.number
                if number >= border:
                    new_chords[0].append(pitch)
                else:
                    new_chords[1].append(pitch + 20)
            for i, newc in enumerate(new_chords):
                voices[i].append(newc)
        return tuple(voices)

    @staticmethod
    def convert_tone_sublists2actual_chord_objects(chords) -> tuple:
        def transform(chord):
            pitches = [t.written_pitch.number for t in chord]
            duration = chord[0].written_duration
            return abjad.Chord(pitches, duration)

        return tuple(transform(chord) for chord in chords)

    @staticmethod
    def mk_fret_change_mark(string_number, new_fret, old_fret_abs, new_fret_abs) -> str:
        diff = new_fret_abs - old_fret_abs
        assert diff != 0
        if diff <= 0:
            fret_change = "-{0}|".format(abs(diff)) + new_fret
        else:
            fret_change = "+{0}|".format(abs(diff)) + new_fret
        if new_fret_abs % 10 != 0 and new_fret_abs % 2 == 0:
            color = r"\with-color #red"
        elif new_fret_abs % 2 != 0 and new_fret_abs % 5 != 0:
            color = r"\with-color #blue"
        else:
            color = ""
        boxes = r'\box \bold "{0}" \hspace #-0.4 {1} \box \caps "{2}"'.format(
            string_number + 1, color, fret_change
        )
        return r"\line{" + boxes + "}"

    @staticmethod
    def mk_fret_change_markup(fret_changes: list) -> abjad.Markup:
        data = r"\column{" + " ".join(fret_changes) + "}"
        return abjad.Markup(data, direction=abjad.OrdinalConstant("y", 1, "Up"))

    @staticmethod
    def mk_glissando_up():
        pass

    @staticmethod
    def mk_glissando_down():
        pass

    @staticmethod
    def mk_vibrato():
        pass

    @staticmethod
    def convert_string_number2pitch_number(string_number: int) -> int:
        # map 21 strings to c major / a minor scale
        mapping = {
            10: 2,
            11: 3,
            12: 5,
            13: 7,
            14: 9,
            15: 10,
            16: 12,
            17: 14,
            18: 15,
            19: 17,
            20: 19,
            9: -3,
            8: -5,
            7: -6,
            6: -8,
            5: -10,
            4: -11,
            3: -13,
            2: -15,
            1: -17,
            0: -18,
        }
        pitch_number = mapping[string_number]
        pitch_number -= (
            Monochord.transposition_percussion_clef
        )  # since you use a percurssion clef
        return pitch_number

    def convert2abjad_pitches(self, pitches) -> list:
        res = []
        for pitch in pitches:
            chord = []
            for p in pitch:
                if p != mel.EmptyPitch():
                    res_string = None
                    for s in self.strings:
                        if p == s.pitch:
                            res_string = s.number
                            break

                    if res_string is not None:
                        pitch_number = Monochord.convert_string_number2pitch_number(
                            res_string
                        )
                        abjad_pitch = abjad.NumberedPitch(pitch_number)
                        chord.append(abjad_pitch)
                    else:
                        raise ValueError(
                            "Pitch: {0} could not be found in Monochord-Tuning!".format(
                                p
                            )
                        )
            res.append(chord)
        return res

    @staticmethod
    def convert_voices2staff(voices: list) -> abjad.StaffGroup:
        # voices contain 2-element tuples (for each staff one subvoice)
        voices = tuple(zip(*voices))
        """
        new_voices = [abjad.Voice([]), abjad.Voice([])]
        for idx, voxs in enumerate(voices):
            for voice in voxs:
                new_voices[idx].extend(voice[:])
        """
        s0 = Instrument.mk_staff(voices[0])
        s1 = Instrument.mk_staff(voices[1])
        # s0 = Instrument.mk_staff([new_voices[0]])
        # s1 = Instrument.mk_staff([new_voices[1]])
        staffgroup = abjad.StaffGroup([s0, s1], context_name="PianoStaff")
        return staffgroup


Monochord.fret_distances = Monochord.calculate_fret_distances()


class Guitar(Instrument):
    def __init__(self, n_divisions=48, octaves=3):
        float_and_oct = tuple(
            (
                n // n_divisions,
                (n % n_divisions) / n_divisions * 12,
                2 ** (n // n_divisions),
            )
            for n in range(n_divisions * octaves)
        )
        float_and_oct += tuple(
            (
                n // n_divisions,
                (n % n_divisions) / n_divisions * 12,
                2 ** (n // n_divisions),
            )
            for n in range(-18, 0)
        )
        self.available_pitches = tuple(
            sorted(
                tuple(
                    (edo.EDO2_12Pitch(fl, oc), abjad.NumberedPitch(fl - 15 + (12 * n)))
                    for n, fl, oc in float_and_oct
                ),
                key=operator.itemgetter(0),
            )
        )

        self.cents_of_available_pitches = tuple(
            p[0].cents for p in self.available_pitches
        )

    def find_suitable_pitch(self, pitch: ji.JIPitch) -> ji.JIPitch:
        lcents_ap = len(self.cents_of_available_pitches)
        ct = pitch.cents
        if ct in self.cents_of_available_pitches:
            idx = self.cents_of_available_pitches.index(ct)
        else:
            closest = bisect.bisect_right(self.cents_of_available_pitches, ct)
            if closest == lcents_ap:
                idx = closest - 1
            elif closest in (0, lcents_ap - 1):
                idx = closest
            else:
                differences = tuple(
                    abs(ct - c)
                    for c in (
                        self.cents_of_available_pitches[closest - 1],
                        self.cents_of_available_pitches[closest],
                        self.cents_of_available_pitches[closest + 1],
                    )
                )
                idx = (differences.index(min(differences)) - 1) + closest

        return self.available_pitches[idx]

    def convert2abjad_voices(self, cadence, grid) -> list:
        def tie_pauses(harmonies, delays):
            newh = []
            newd = []
            first = True
            for harmony, delay in zip(harmonies, delays):
                if not harmony and first is False:
                    newd[-1] += delay
                else:
                    newh.append(harmony)
                    newd.append(delay)
                first = False
            return newh, newd

        cadence = cadence.tie_pauses()
        pitches_and_delays = tie_pauses(
            self.convert2abjad_pitches(cadence.pitch), cadence.delay
        )
        voice = self.convert_abjad_pitches_and_mu_rhythms2abjad_notes(
            pitches_and_delays[0], pitches_and_delays[1], grid
        )
        return [voice]

    def convert2abjad_pitches(self, pitches) -> list:
        res = []
        for pitch in pitches:
            chord = []
            for p in pitch:
                if p != mel.TheEmptyPitch:
                    chord.append(self.find_suitable_pitch(p)[1])
            res.append(chord)
        return res

    @staticmethod
    def convert_voices2staff(voices: list, clef="treble") -> abjad.Staff:
        return Guitar.mk_staff(voices, clef)
