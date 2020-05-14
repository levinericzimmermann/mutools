import functools
import operator
import quicktions as fractions
import subprocess

import abjad

import xml.etree.ElementTree as ET

from mu.mel import mel
from mu.midiplug import midiplug
from mu.sco import old
from mu.rhy import rhy
from mu.utils import tools

from . import bpm_extract
from . import lily
from . import quantizise


class Transcription(object):
    r"""Class for the transcription of monophonic melodies.

    First melodies have to be analysed with the software Tony. Then the results have to
    be saved in the SVL format. Those files can be further analysed and transformed
    using this Transcription class.

    Example usage:

    >>> from mu.mel import ji
    >>> # depending on the particular definition of the scale, the transcription is more
    >>> # precise or less precise (more different from the original)
    >>> scale = (
    >>>     ji.r(1, 1),
    >>>     ji.r(9, 8),
    >>>     ji.r(5, 4),
    >>>     ji.r(11, 8),
    >>>     ji.r(16, 11),
    >>>     ji.r(32, 17),
    >>> )
    >>> # using 5/4 as the tonic of the scale
    >>> scale = tuple(sorted((p - ji.r(5, 4)).normalize() for p in scale))
    >>> trans0 = Transcription.from_scale(
    >>>     "qiroah0",
    >>>     scale,
    >>>     scale_degree_of_first_pitch=1,
    >>>     tolerance_factor_for_same_scale_degree=None,
    >>> )
    >>> n_divisions = 8
    >>> min_tone_size = fractions.Fraction(1, 32)
    >>> min_rest_size = fractions.Fraction(1, 16)
    >>> stretch_factor = fractions.Fraction(1, 8)
    >>> trans0.convert2score(
    >>>     reference_pitch=5,  # F is reference pitch
    >>>     min_tone_size=min_tone_size,
    >>>     min_rest_size=min_rest_size,
    >>>     stretch_factor=stretch_factor,
    >>> )
    >>> trans0.synthesize(
    >>>     concert_pitch=260 * 4 / 3,
    >>>     stretch_factor=stretch_factor,
    >>>     min_tone_size=min_tone_size,
    >>>     min_rest_size=min_rest_size,
    >>> )
    """

    def __init__(self, name: str, tones: tuple, frequency_range: tuple = None):
        self.__name = name
        self.__tones = tones
        self.__frequency_range = frequency_range

    @property
    def name(self) -> str:
        return self.__name

    @property
    def tones(self) -> tuple:
        return self.__tones

    @property
    def frequency_range(self) -> tuple:
        return self.__frequency_range

    @staticmethod
    def _get_root(path: str):
        tree = ET.parse(path)
        return tree.getroot()

    @staticmethod
    def _filter_data_from_root(root) -> tuple:
        data = []
        sr = int(root[0][0].attrib["sampleRate"])
        for child in root[0][1]:
            start = int(child.attrib["frame"]) / sr
            duration = int(child.attrib["duration"]) / sr
            freq = float(child.attrib["value"])
            volume = float(child.attrib["level"])
            stop_time = start + duration

            data.append([freq, start, stop_time, volume])

        return tuple(map(tuple, data))

    @staticmethod
    def estimate_tempo(
        name: str, method: str = "essentia", params: dict = None
    ) -> float:
        return bpm_extract.BPM("{}.wav".format(name), method=method, params=params)

    @staticmethod
    def _convert_data2melody(
        data: tuple, name: str, tempo_estimation_method: str = "essentia"
    ) -> tuple:

        tempo = Transcription.estimate_tempo(name, method=tempo_estimation_method)
        melody = old.Melody([], time_measure="absolute")

        for tone in data:
            pitch, start, stop_time, volume = tone

            if melody:
                stop_last_tone = melody[-1].duration
                difference = start - stop_last_tone

                if difference > 0:
                    melody.append(old.Tone(mel.TheEmptyPitch, stop_last_tone, start))

                elif difference < 0:
                    melody[-1].duration += difference

            else:
                if start != 0:
                    melody.append(old.Tone(mel.TheEmptyPitch, 0, start))

            melody.append(old.Tone(pitch, start, stop_time, volume=volume))

        melody = melody.convert2relative()

        if melody[0].pitch.is_empty:
            melody = melody[1:]

        factor = tempo / 60

        melody.delay = tuple(d * factor for d in melody.delay)
        melody.dur = tuple(d * factor for d in melody.dur)

        return tuple(melody)

    @classmethod
    def from_scale(
        cls,
        name: str,
        scale: tuple,
        scale_degree_of_first_pitch: int = 0,
        octave_of_first_pitch: int = 0,
        tolerance_factor_for_same_scale_degree: float = 0.9,
        tempo_estimation_method: str = "essentia",
    ) -> "Transcription":
        def normalize_scale_degree(scale_degree: int, octave: int) -> tuple:
            if scale_degree == scale_size:
                scale_degree = 0
                octave += 1

            elif scale_degree == -1:
                scale_degree = scale_size - 1
                octave -= 1

            return scale_degree, octave

        # testing scale for correct values
        for p in scale:
            assert p.octave == 0

        # sorting scale
        scale = tuple(sorted(scale))

        scale_size = len(scale)

        assert all(
            (scale_degree_of_first_pitch >= 0, scale_degree_of_first_pitch < scale_size)
        )

        root = cls._get_root("{}.svl".format(name))
        frequency_range = root[0][0].attrib["minimum"], root[0][0].attrib["maximum"]
        data = cls._filter_data_from_root(root)
        new_data = [(scale_degree_of_first_pitch, octave_of_first_pitch) + data[0][1:]]

        for idx, tone in enumerate(data[1:]):
            ct_difference = mel.SimplePitch.hz2ct(data[idx][0], tone[0])
            last_scale_degree, last_octave, *_ = new_data[-1]
            last_pitch = scale[last_scale_degree].register(last_octave)
            last_pitch_cents = last_pitch.cents

            if ct_difference > 0:
                operation = operator.add
                inverse_operation = operator.sub
                comparision = operator.ge

            else:
                operation = operator.sub
                inverse_operation = operator.add
                comparision = operator.le

            next_scale_degree, next_octave = map(int, (last_scale_degree, last_octave))
            while True:
                next_scale_degree = operation(next_scale_degree, 1)
                next_scale_degree, next_octave = normalize_scale_degree(
                    next_scale_degree, next_octave
                )

                current_pitch = scale[next_scale_degree].register(next_octave)
                current_ct_difference = current_pitch.cents - last_pitch_cents

                if comparision(current_ct_difference, ct_difference):
                    break

            options = (
                normalize_scale_degree(
                    inverse_operation(next_scale_degree, 1), next_octave
                ),
                (next_scale_degree, next_octave),
            )

            if (
                options[0] == (last_scale_degree, last_octave)
                and tolerance_factor_for_same_scale_degree
            ):
                tolerance = float(tolerance_factor_for_same_scale_degree)
            else:
                tolerance = None

            options_as_pitches = tuple(
                map(lambda data: scale[data[0]].register(data[1]), options)
            )
            difference_to_expected_distance = [
                abs(ct_difference - (p.cents - last_pitch_cents))
                for p in options_as_pitches
            ]

            if tolerance is not None:
                difference_to_expected_distance[1] *= tolerance

            if difference_to_expected_distance[1] <= difference_to_expected_distance[0]:
                choosen_pitch = options[1]
            else:
                choosen_pitch = options[0]

            new_data.append(choosen_pitch + tone[1:])

        new_data = tuple((scale[d[0]].register(d[1]),) + d[2:] for d in new_data)
        melody = cls._convert_data2melody(new_data, name, tempo_estimation_method)
        return cls(name, tuple(melody), frequency_range)

    @classmethod
    def from_concert_pitch(
        cls,
        name: str,
        concert_pitch: float = None,
        tempo_estimation_method: str = "essentia",
    ) -> "Transcription":

        root = cls._get_root("{}.svl".format(name))
        frequency_range = root[0][0].attrib["minimum"], root[0][0].attrib["maximum"]
        data = cls._filter_data_from_root(root)

        if concert_pitch is None:
            concert_pitch = data[0][0]

        new_data = []
        for tone in data:
            freq = tone[0]
            cents = mel.SimplePitch.hz2ct(concert_pitch, freq)
            pitch = mel.SimplePitch(concert_pitch_freq=concert_pitch, cents=cents)
            new_data.append((pitch,) + tone[1:])

        melody = cls._convert_data2melody(new_data, name, tempo_estimation_method)
        return cls(name, tuple(melody), frequency_range)

    def quantizise(
        self,
        stretch_factor: float = 1,
        n_divisions: int = 8,
        min_tone_size: fractions.Fraction = 0,
        min_rest_size: fractions.Fraction = fractions.Fraction(1, 10),
    ) -> tuple:
        delays = rhy.Compound(tuple(t.delay for t in self.tones)).stretch(
            stretch_factor
        )
        tones = old.Melody(self.tones).copy()
        tones.delay = delays
        tones.dur = delays

        melody = quantizise.quantisize_rhythm(
            tones, n_divisions, min_tone_size, min_rest_size
        )
        return melody.pitch, melody.delay

    def convert2score(
        self,
        reference_pitch: int = 0,
        stretch_factor: float = 1,
        n_divisions: int = 8,
        min_tone_size: fractions.Fraction = 0,
        min_rest_size: fractions.Fraction = fractions.Fraction(1, 10),
    ) -> None:

        pitches, delays = self.quantizise(
            stretch_factor=stretch_factor,
            n_divisions=n_divisions,
            min_tone_size=min_tone_size,
            min_rest_size=min_rest_size,
        )

        bar_grid = tuple(fractions.Fraction(1, 1) for i in range(15))
        grid = tuple(fractions.Fraction(1, 4) for i in range(50))

        notes = abjad.Voice([])

        absolute_delay = tools.accumulate_from_zero(delays)
        for pitch, delay, start, stop in zip(
            pitches, delays, absolute_delay, absolute_delay[1:]
        ):
            seperated_by_bar = tools.accumulate_from_n(
                lily.seperate_by_grid(start, stop, bar_grid, hard_cut=True), start
            )
            sub_delays = functools.reduce(
                operator.add,
                tuple(
                    functools.reduce(
                        operator.add,
                        tuple(
                            lily.seperate_by_assignability(d)
                            for d in lily.seperate_by_grid(start, stop, grid)
                        ),
                    )
                    for start, stop in zip(seperated_by_bar, seperated_by_bar[1:])
                ),
            )
            subnotes = []
            if pitch.is_empty:
                ct = None
            else:
                ct = pitch.cents / 100
                # round to 12th tone
                ct = round(ct * 6) / 6
                ct += reference_pitch

            for delay in sub_delays:
                if ct is None:
                    obj = abjad.Rest(delay)
                else:
                    obj = abjad.Note(ct, delay)

                subnotes.append(obj)

            if ct is not None and len(subnotes) > 1:
                for note in subnotes[:-1]:
                    abjad.attach(abjad.Tie(), note)

            notes.extend(subnotes)

        score = abjad.Score([notes])

        with open("{}.ly".format(self.name), "w") as f:
            f.write('\\version "2.19.83"\n')
            f.write(lily.EKMELILY_PREAMBLE)
            f.write("\n")
            f.write(format(score))

        subprocess.call(["lilypond", "{}.ly".format(self.name)])

    def synthesize(
        self,
        concert_pitch: float = 260,
        stretch_factor: float = 1,
        n_divisions: int = 8,
        min_tone_size: fractions.Fraction = 0,
        min_rest_size: fractions.Fraction = fractions.Fraction(1, 10),
    ) -> None:
        pitches, delays = self.quantizise(
            stretch_factor=stretch_factor,
            n_divisions=n_divisions,
            min_tone_size=min_tone_size,
            min_rest_size=min_rest_size,
        )

        sequence = []
        for p, d in zip(pitches, delays):
            p.multiply = concert_pitch
            d *= 4
            sequence.append(midiplug.PyteqTone(p, d, d))

        midiplug.Pianoteq(sequence).export2wav(
            "{}_transcription".format(self.name), preset='"Erard Player"'
        )
