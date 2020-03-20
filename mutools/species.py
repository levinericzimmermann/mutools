import abc
import functools
import itertools
import operator

from mu.mel import ji
from mu.mel import mel

from mu.rhy import indispensability

from mu.sco import old

from mu.utils import prime_factors
from mu.utils import tools

from mu.midiplug import midiplug

from pbIII.fragments import counterpoint
from pbIII.soil import soil

"""
class __MetaSpecies(object):
    def __new__(cls, name, bases, attrs):
        def auto_init(self, *args, **kwargs):
            arg_names = cls.tone_args + tuple(self._init_args.keys())
            length_tone_args = len(cls.tone_args)
            length_args = len(args)
            for counter, arg_val, arg_name in zip(range(length_args), args, arg_names):
                if counter > length_tone_args:
                    tolerance = self.__init_args[arg_name][0]
                    try:
                        assert arg_val <= tolerance[0]
                        assert arg_val >= tolerance[1]
                    except AssertionError:
                        msg = "The value for '{0}' has to be >= {1} and <= {2}.".format(
                            arg_name, tolerance[0], tolerance[1]
                        )
                        raise ValueError(msg)
                setattr(self, arg_name, arg_val)

            for arg_name in arg_names[length_args:]:
                if arg_name not in kwargs.keys():
                    kwargs.update({arg_name: None})

            self.__dict__.update(kwargs)

            MidiTone.__init__(
                self,
                self.pitch,
                self.delay,
                self.duration,
                self.volume,
                self.glissando,
                self.vibrato,
                self.tuning,
            )

        attrs["__init__"] = auto_init
        return super(SynthesizerMidiTone, cls).__new__(cls, name, bases, attrs)
"""


# class Species(object, metaclass=__MetaSpecies):
class Species(object):
    anatomy = {"gender": {True: {}, False: {}}, None: {"concrete": []}}

    def __init__(
        self, gender, primary_index: int, secondary_index: int, tertiary_index: int
    ) -> None:
        assert gender in (True, False, None)
        self.gender = gender
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.tertiary_index = tertiary_index

    @abc.abstractmethod
    def render(self) -> None:
        raise NotImplementedError


class MonophonicSpecies(Species):
    """Abstract class for Species that can't handle mixed gender as input."""

    def __init__(
        self, gender, primary_index: int, secondary_index: int, tertiary_index: int
    ) -> None:
        if gender is None:
            raise NotImplementedError("This species can't handle mixed gender.")
        super(MonophonicSpecies, self).__init__(
            self, gender, primary_index, secondary_index, tertiary_index
        )


class DualisticSpecies(Species):
    """Abstract class for Species that can't handle only one gender as input."""

    def __init__(
        self, gender, primary_index: int, secondary_index: int, tertiary_index: int
    ) -> None:
        if gender is not None:
            raise NotImplementedError("This species can't handle only one gender.")
        super(DualisticSpecies, self).__init__(
            self, gender, primary_index, secondary_index, tertiary_index
        )


class FreeStyleCP(MonophonicSpecies):
    n_attacks_cycle = itertools.cycle((2, 3, 4))
    max_spectrum_profile_change = 5  # dB

    def __init__(
        self,
        groups: tuple,
        n_bars_per_group: tuple,
        register_per_voice: tuple,
        tempo: float,
        rhythmic_mode: int = 0,
        tonic_on_last_bar: bool = True,
        constraints: tuple = tuple([]),
        add_melisma: bool = False,
        start_triad: tuple = None,
        humanized: float = 10,
        dynamic_range: tuple = (0.1, 0.85),
    ) -> None:

        # there are 6 different rhythmic modes.
        # they are indicating with which rhythmic
        # permutation the counterpoint will start.
        assert rhythmic_mode in tuple(range(6))

        self.dynamic_range = dynamic_range
        self.absolute_dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        self.tempo = tempo
        self.groups = groups
        self.tonic_on_last_bar = tonic_on_last_bar
        self.n_bars_per_group = n_bars_per_group

        if not constraints:
            constraints = self.make_standard_constraints()

        self.group_indices = tuple(
            soil.MALE.detect_group_index(group[1:]) for group in groups
        )
        self.metre_per_vox_per_group = tuple(
            soil.MALE.metre_per_vox_per_bar[idx] for idx in self.group_indices
        )
        self.n_bars = sum(n_bars_per_group)
        self.group_index_per_bar = functools.reduce(
            operator.add,
            tuple(
                tuple(idx for n in range(n_bars))
                for idx, n_bars in enumerate(n_bars_per_group)
            ),
        )
        self.rhythmic_mode = rhythmic_mode

        # (1) make rhythms
        ry, ind = self.make_rhythm_per_voice()
        self.rhythms_per_voice = ry
        self.indispensability_per_voice = ind

        # (2) make main pitches
        self.harmonic_frame = counterpoint.HarmonicFrame(
            self.rhythms_per_voice,
            groups,
            n_bars_per_group,
            start_triad=start_triad,
            constraints=constraints,
        )
        self.frame_pitches = self.harmonic_frame()
        self.frame_pitches_per_voice = tuple(
            self.find_best_voice_leading_per_voice(
                self.delete_repeating_pitches_in_frame_pitches(vox), voice_range
            )
            for vox, voice_range in zip(
                tuple(zip(*self.frame_pitches)), register_per_voice
            )
        )

        self.instruments_per_voice = soil.MALE.instrument_per_vox_per_bar[
            self.group_indices[0]
        ]

        self.real_frame_rhythms_per_voice = tuple(
            soil.MALE.convert2real_rhythm(voice, duration=self.tempo)[0]
            for voice in self.rhythms_per_voice
        )
        # self.real_frame_rhythms_per_voice = self.harmonic_frame.relative_attacks
        # print(self.real_frame_rhythms_per_voice)

        self.frame_voices = tuple(
            tuple(old.Tone(p, r) for p, r in zip(pitches, rhythms))
            for pitches, rhythms in zip(
                self.frame_pitches_per_voice, self.real_frame_rhythms_per_voice
            )
        )

        iadp = self.detect_intervals_and_duration_pairs_per_tone_per_voice(
            self.frame_voices
        )
        self.intervals_and_duration_pairs = iadp
        hatpptpv = self.detect_harmonicity_per_tone_per_voice(
            self.intervals_and_duration_pairs
        )
        self.harmonicity_per_tone_per_voice = hatpptpv
        self.dynamic_per_tone_per_voice = self.calculate_dynamics_for_frame_pitches()
        self.spectrum_profile_per_voice = self.detect_spectrum_profile_per_tone(
            self.intervals_and_duration_pairs, self.max_spectrum_profile_change
        )
        self.glissando_per_voice = tuple(
            self.make_glissandi(vox) for vox in self.frame_voices
        )

        # (3) add additional pitches

    @staticmethod
    def mk_empty_attack(duration: float, volume: float) -> midiplug.PyteqTone:
        return midiplug.PyteqTone(
            ji.JIPitch(ji.r(1, 4), multiply=soil.CONCERT_PITCH),
            duration,
            duration,
            volume=volume * 0.75,
            hammer_noise=3,
            impedance=0.3,
            cutoff=0.3,
            q_factor=5,
            string_length=0.8,
            strike_point=1 / 2,
            # hammer_hard_piano=0.4,
            # hammer_hard_mezzo=0.8,
            # hammer_hard_forte=1,
            # blooming_energy=0,
        )

    def render(self) -> None:
        real_rhythms_per_voice = self.real_frame_rhythms_per_voice
        for (
            v_idx,
            pitches,
            rhythms,
            volumes,
            instr,
            glissando_per_tone,
            spectrum_profile_per_tone,
        ) in zip(
            range(3),
            self.frame_pitches_per_voice,
            real_rhythms_per_voice,
            self.dynamic_per_tone_per_voice,
            self.instruments_per_voice,
            self.glissando_per_voice,
            self.spectrum_profile_per_voice,
        ):

            # blooming_energy_cycle = itertools.cycle(
            #   (0, 0.2, 0, 0.5, 0.6, 0, 0.4, 0.2))

            if instr[0] in (
                "Celtic Harp Bright",
                "Concert Harp Daily",
                "Glockenspiel Humanized no stretching",
            ):
                sustain_pedal = 1
            else:
                sustain_pedal = 0

            import random

            sequence = tuple(
                self.mk_empty_attack(r, v)
                if p.is_empty
                else midiplug.PyteqTone(
                    ji.JIPitch(p, multiply=soil.CONCERT_PITCH),
                    r,
                    r,
                    volume=v,
                    glissando=glissando,
                    sustain_pedal=sustain_pedal,
                    hammer_noise=1,
                    impedance=random.uniform(2.25, 3),
                    # cutoff=1.5,
                    # q_factor=3,
                    cutoff=3,
                    q_factor=0.2,
                    # string_length=10,
                    strike_point=1 / 8,
                    # hammer_hard_piano=0.1,
                    # hammer_hard_mezzo=0.65,
                    # hammer_hard_forte=1.3,
                    spectrum_profile_3=spectrum_profile[0],
                    spectrum_profile_5=spectrum_profile[1],
                    spectrum_profile_6=spectrum_profile[0],
                    spectrum_profile_7=spectrum_profile[2],
                    # blooming_energy=next(blooming_energy_cycle),
                    # blooming_energy=0,
                )
                for p, r, v, glissando, spectrum_profile in zip(
                    pitches,
                    rhythms,
                    volumes,
                    glissando_per_tone,
                    spectrum_profile_per_tone,
                )
            )

            pteq = midiplug.Pianoteq(sequence)

            if instr[1]:
                preset = None
                fxp = '"pbIII/fxp/{0}.fxp"'.format(instr[0])
            else:
                preset = '"{0}"'.format(instr[0])
                fxp = None

            pteq.export2wav("test{0}".format(v_idx), preset=preset, fxp=fxp)

        for v_idx, voice, instr in zip(
            range(3), self.mk_metrum_for_every_voice(), self.instruments_per_voice
        ):
            if instr[0] in (
                "Celtic Harp Bright",
                "Concert Harp Daily",
                "Glockenspiel Humanized no stretching",
            ):
                sustain_pedal = 1
            else:
                sustain_pedal = 0

            sequence = tuple(self.mk_empty_attack(r, 0.3) for r in voice)
            pteq = midiplug.Pianoteq(sequence)

            if instr[1]:
                preset = None
                fxp = '"pbIII/fxp/{0}.fxp"'.format(instr[0])
            else:
                preset = '"{0}"'.format(instr[0])
                fxp = None

            pteq.export2wav("metrum{0}".format(v_idx), preset=preset, fxp=fxp)

    def mk_metrum_for_every_voice(self) -> tuple:
        voices = []
        for v_idx in range(3):
            meters_per_group = tuple(
                soil.MALE.metre_per_vox_per_bar[bar_idx][v_idx]
                for bar_idx in self.group_indices
            )
            metrum_per_group = []
            for meter in meters_per_group:
                metrum_per_group.append(tuple(1 for i in range(meter)))
            voice = functools.reduce(
                operator.add,
                tuple(
                    tuple(metrum for n in range(n_bars))
                    for n_bars, metrum in zip(self.n_bars_per_group, metrum_per_group)
                ),
            )
            voice = soil.MALE.convert2real_rhythm(voice, duration=self.tempo)[0]
            voices.append(voice)
        return tuple(voices)

    @staticmethod
    def make_glissandi(voice: tuple) -> tuple:
        # first finding glissandi for interpolation
        glissando_size_per_interpolation = [0]
        for t0, t1 in zip(voice, voice[1:]):
            if not any((t0.pitch.is_empty, t1.pitch.is_empty)):
                distance = t1.pitch - t0.pitch
                cent_distance = distance.cents
                if abs(cent_distance) < 1200:
                    cent_distance = 0
                else:
                    cent_distance -= 50
            else:
                if t1.pitch.is_empty and not t0.pitch.is_empty:
                    cent_distance = 0
                else:
                    cent_distance = 0
            glissando_size_per_interpolation.append(cent_distance)

        glissando_size_per_ornamentation = []
        voice_size = len(voice)
        direction_cycle = itertools.cycle((True, False))
        sorted_duration_per_tone = sorted(
            tuple((t_idx, float(t.delay)) for t_idx, t in enumerate(voice)),
            key=operator.itemgetter(1),
            reverse=True,
        )
        tone_idx_for_tones_with_ornamentation = tuple(
            item[0] for item in sorted_duration_per_tone[: int(voice_size * 0)]
        )
        # then adding glissandi for ornamentation at the beginning
        for t_idx, t, glissando_size in zip(
            range(voice_size), voice, glissando_size_per_interpolation
        ):

            if t_idx in tone_idx_for_tones_with_ornamentation:
                if glissando_size != 0:
                    ornamentaton_direction = glissando_size < 0
                else:
                    ornamentaton_direction = next(direction_cycle)
                ornamentaton_size = 100
                if not ornamentaton_direction:
                    ornamentaton_size *= -1
            else:
                ornamentaton_size = 0

            glissando_size_per_ornamentation.append(ornamentaton_size)

        glissando_size_per_interpolation = glissando_size_per_interpolation[1:]
        glissando_size_per_interpolation += [0]

        # combining both and making proper glissandi objects
        standard_ornamentation_duration = 0.3
        glissando_per_tone = []
        for tone, ornamentation_size, glissando_size in zip(
            voice, glissando_size_per_ornamentation, glissando_size_per_interpolation
        ):
            tone_duration = float(tone.duration)
            ornamentation_duration = standard_ornamentation_duration
            if tone_duration < ornamentation_duration * 4:
                ornamentation_duration = tone_duration * 0.2

            interpolation_duration = (tone_duration - ornamentation_duration) * 0.3
            stable_duration = (
                tone_duration - interpolation_duration - ornamentation_duration
            )

            pitch_line = old.InterpolationLine(
                old.PitchInterpolation(
                    duration, mel.SimplePitch(soil.CONCERT_PITCH, size)
                )
                for size, duration in zip(
                    (ornamentaton_size, 0, 0, glissando_size),
                    (
                        ornamentation_duration,
                        stable_duration,
                        interpolation_duration,
                        0,
                    ),
                )
            )
            glissando = old.GlissandoLine(pitch_line)
            glissando_per_tone.append(glissando)

        return tuple(glissando_per_tone)

    @staticmethod
    def detect_spectrum_profile_per_tone(
        intervals_and_duration_pairs: tuple, max_spectrum_profile_change: float
    ) -> tuple:
        spectrum_profile_per_voice = []
        ig1 = operator.itemgetter(1)

        for voice in intervals_and_duration_pairs:
            prime_counter_per_tone = []
            for tone in voice:
                primes_and_values = [0, 0, 0]
                complete_duration = float(sum(ig1(itp) for itp in tone))
                for interval_time_pair in tone:
                    vector = interval_time_pair[0].monzo
                    while len(vector) < 4:
                        vector += (0,)
                    percentage = float(interval_time_pair[1] / complete_duration)
                    for pidx, prime_exponent in enumerate(vector[1:4]):
                        primes_and_values[pidx] += percentage * prime_exponent
                prime_counter_per_tone.append(tuple(primes_and_values))

            minima_and_maxima_per_prime = tuple(
                (min(p), max(p)) for p in zip(*prime_counter_per_tone)
            )
            range_per_prime = tuple(ma - mi for mi, ma in minima_and_maxima_per_prime)

            dp_minima_and_maxima_per_prime = []
            for minima_and_maxima in minima_and_maxima_per_prime:
                if minima_and_maxima[0] < 0:
                    minima = -max_spectrum_profile_change
                else:
                    minima = 0

                if minima_and_maxima[1] > 0:
                    maxima = max_spectrum_profile_change
                else:
                    maxima = 0

                dp_minima_and_maxima_per_prime.append((minima, maxima))

            dp_range_per_prime = tuple(
                dp_minima_and_maxima[1] - dp_minima_and_maxima[0]
                for dp_minima_and_maxima in dp_minima_and_maxima_per_prime
            )
            spectrum_profile_per_tone = tuple(
                tuple(
                    (dp_range * ((p - min_and_max[0]) / total_range)) + dp_min_max[0]
                    if total_range
                    else 0
                    for p, min_and_max, total_range, dp_min_max, dp_range in zip(
                        primes,
                        minima_and_maxima_per_prime,
                        range_per_prime,
                        dp_minima_and_maxima_per_prime,
                        dp_range_per_prime,
                    )
                )
                for primes in prime_counter_per_tone
            )
            spectrum_profile_per_voice.append(tuple(spectrum_profile_per_tone))

        return tuple(spectrum_profile_per_voice)

    @staticmethod
    def detect_intervals_and_duration_pairs_per_tone_per_voice(voices: tuple) -> tuple:
        voices = tuple(old.Melody(v) for v in voices)
        interval_and_time_pairs_per_tone_per_voice = []
        for v_idx, voice in enumerate(voices):
            poly = old.Polyphon(v for v_idx1, v in enumerate(voices) if v_idx1 != v_idx)
            melody = voice.copy()
            interval_and_time_pairs_per_tone = []
            for tone in melody.convert2absolute_time():
                start, stop = tone.delay, tone.duration
                simultan_events = functools.reduce(
                    operator.add, tuple(m[:] for m in poly.cut_up_by_time(start, stop))
                )
                interval_and_time_pairs = []
                for simultan_event in simultan_events:
                    if not simultan_event.pitch.is_empty:
                        interval = simultan_event.pitch - tone.pitch
                        interval_and_time_pairs.append(
                            (interval, simultan_event.duration)
                        )
                interval_and_time_pairs_per_tone.append(tuple(interval_and_time_pairs))
            interval_and_time_pairs_per_tone_per_voice.append(
                tuple(interval_and_time_pairs_per_tone)
            )
        return tuple(interval_and_time_pairs_per_tone_per_voice)

    @staticmethod
    def detect_harmonicity_and_time_pairs_per_tone_per_voice(
        interval_and_time_pairs_per_tone_per_voice: tuple
    ) -> tuple:
        return tuple(
            tuple(
                tuple((pair[0].harmonicity_simplified_barlow, pair[1]) for pair in tone)
                for tone in voice
            )
            for voice in interval_and_time_pairs_per_tone_per_voice
        )

    def detect_harmonicity_per_tone_per_voice(
        self, interval_and_time_pairs_per_tone_per_voice: tuple
    ) -> tuple:
        hatpptpv = self.detect_harmonicity_and_time_pairs_per_tone_per_voice(
            interval_and_time_pairs_per_tone_per_voice
        )
        harmonicity_per_tone_per_voice = []
        ig1 = operator.itemgetter(1)
        for voice in hatpptpv:
            newv = []
            for tone in voice:
                durations = tuple(ig1(s) for s in tone)
                complete_duration = sum(durations)
                newv.append(sum(h * (d / complete_duration) for h, d in tone))
            harmonicity_per_tone_per_voice.append(tuple(newv))
        return tuple(harmonicity_per_tone_per_voice)

    def calculate_dynamics_for_frame_pitches(self) -> tuple:
        absolute_range = self.absolute_dynamic_range
        min_vol = self.dynamic_range[0]
        vol_per_voice = []
        for metric_strength_per_tone, harmonicity_per_tone in zip(
            self.indispensability_per_voice, self.harmonicity_per_tone_per_voice
        ):
            sorted_harmonicity = sorted(set(harmonicity_per_tone))
            n_harmonicities = len(sorted_harmonicity) + 1
            normalized_harmonicity = tuple(
                (sorted_harmonicity.index(h) + 1) / n_harmonicities
                for h in harmonicity_per_tone
            )
            volumes = []
            for harmonicity, metricity in zip(
                normalized_harmonicity, metric_strength_per_tone
            ):
                volume = min_vol + (
                    absolute_range * ((harmonicity * 0.3) + (metricity * 0.7))
                )
                volumes.append(volume)
            vol_per_voice.append(tuple(volumes))
        return tuple(vol_per_voice)

    @staticmethod
    def delete_repeating_pitches_in_frame_pitches(voice: tuple):
        new_voice = [voice[0]]
        for p in voice[1:]:
            if new_voice[-1] != p:
                new_voice.append(p)
        return tuple(new_voice)

    @staticmethod
    def find_best_voice_leading_per_voice(voice: tuple, voice_range: tuple) -> tuple:
        splited = tools.split_iterable_by_function(voice, lambda p: p.is_empty)
        res = []
        for part in splited:
            if part[-1].is_empty:
                part = part[:-1]
                has_rest = True
            else:
                has_rest = False

            res.extend(ji.find_best_voice_leading(part, voice_range))
            if has_rest:
                res.append(mel.TheEmptyPitch)
        return tuple(res)

    def make_standard_constraints(self) -> tuple:
        constraints = (counterpoint.HFC_forbid_too_empty_harmonies,)

        if self.tonic_on_last_bar:
            constraints += (counterpoint.HFC_assert_last_triad_is_tonic,)

        return constraints

    def detect_n_attacks_per_bar_per_voice(self) -> tuple:
        allowed_amount_of_attacks_per_bar = (3, 3, 3)
        permutations = tuple(itertools.permutations(allowed_amount_of_attacks_per_bar))
        cycle = [permutations[self.rhythmic_mode]]
        for n in range(2):
            last_per = cycle[-1]
            new_per = []
            for item in last_per:
                idx = allowed_amount_of_attacks_per_bar.index(item)
                if idx + 1 == len(allowed_amount_of_attacks_per_bar):
                    new_item = allowed_amount_of_attacks_per_bar[0]
                else:
                    new_item = allowed_amount_of_attacks_per_bar[idx + 1]
                new_per.append(new_item)
            cycle.append(tuple(new_per))
        cycle = itertools.cycle(cycle)
        n_attacks_per_voice_per_bar = list(next(cycle) for n in range(self.n_bars - 1))
        if self.tonic_on_last_bar:
            n_attacks_per_voice_per_bar.append((1, 1, 1))
        else:
            n_attacks_per_voice_per_bar.append(next(cycle))

        return tuple(zip(*n_attacks_per_voice_per_bar))

    def detect_n_attacks_on_first_beat_of_the_bar(self) -> tuple:
        cycle = itertools.cycle((2, 1))
        n_attacks = [3]
        for m in range(self.n_bars - 2):
            n_attacks.append(next(cycle))

        if self.tonic_on_last_bar:
            n_attacks.append(3)
        else:
            n_attacks.append(next(cycle))

        return tuple(n_attacks)

    def distribute_n_first_beat_attacks_on_voices(
        self, n_attacks_on_first_beat_of_the_bar_per_bar: tuple
    ) -> tuple:
        voices = (0, 1, 2)
        permutations = tuple(itertools.permutations(voices))
        cycle = itertools.cycle(
            functools.reduce(
                operator.add,
                permutations[self.rhythmic_mode :] + permutations[: self.rhythmic_mode],
            )
        )
        bars = []
        for (
            n_attacks_on_the_first_beat_of_the_bar
        ) in n_attacks_on_first_beat_of_the_bar_per_bar:
            if n_attacks_on_the_first_beat_of_the_bar == 3:
                played_voices = (0, 1, 2)
            else:
                played_voices = tuple(
                    next(cycle) for n in range(n_attacks_on_the_first_beat_of_the_bar)
                )
            bars.append(tuple(True if v in played_voices else False for v in voices))
        return tuple(zip(*bars))

    @staticmethod
    def make_rhythm(
        metre: int, n_attacks: int, does_contain_attack_on_the_first_beat: bool
    ) -> tuple:

        if metre == 10:
            divided = (2, 5)
        else:
            divided = tuple(prime_factors.factorise(metre))

        indispensability_for_bar = indispensability.indispensability_for_bar(divided)
        ranking = indispensability.bar_indispensability2indices(
            indispensability_for_bar
        )

        if not does_contain_attack_on_the_first_beat:
            ranking = ranking[1:]

        choosen_attacks = sorted(ranking[:n_attacks])

        max_in = max(indispensability_for_bar)
        indispensability_percent = tuple(n / max_in for n in indispensability_for_bar)
        energy_of_choosen_attacks = tuple(
            indispensability_percent[n] for n in choosen_attacks
        )

        relative_attacks = tuple(
            b - a for a, b in zip(choosen_attacks, choosen_attacks[1:] + [metre])
        )

        if not does_contain_attack_on_the_first_beat:
            relative_attacks = ((choosen_attacks[0],),) + relative_attacks

        return relative_attacks, energy_of_choosen_attacks

    def make_rhythm_per_voice(self) -> tuple:
        n_attacks_per_bar_per_voice = self.detect_n_attacks_per_bar_per_voice()
        n_attacks_on_first_beat_of_the_bar = (
            self.detect_n_attacks_on_first_beat_of_the_bar()
        )
        does_contain_first_beat_pv = self.distribute_n_first_beat_attacks_on_voices(
            n_attacks_on_first_beat_of_the_bar
        )

        rhythms_per_voice = []
        indispensability_per_voice = []
        for voice_idx, n_attacks_per_bar, does_contain_first_beat_per_bar in zip(
            range(3), n_attacks_per_bar_per_voice, does_contain_first_beat_pv
        ):
            rhythms = []
            indispensability_percent = []
            for n_attacks, does_contain_first_beat, group_idx in zip(
                n_attacks_per_bar,
                does_contain_first_beat_per_bar,
                self.group_index_per_bar,
            ):
                metre = self.metre_per_vox_per_group[group_idx][voice_idx]
                r, ind = self.make_rhythm(metre, n_attacks, does_contain_first_beat)
                rhythms.append(r)
                indispensability_percent.extend(ind)

            rhythms_per_voice.append(tuple(rhythms))
            indispensability_per_voice.append(tuple(indispensability_percent))

        return tuple(rhythms_per_voice), tuple(indispensability_per_voice)
