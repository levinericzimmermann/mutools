import abc
import bisect
import fractions
import functools
import itertools
import logging
import operator
import os
import subprocess
import time
import uuid

import abjad

import crosstrainer

from mu.mel import ji
from mu.sco import old
from mu.utils import tools

from mutools import attachments
from mutools import lily
from mutools import synthesis


STANDARD_RESOLUTION = 500


def _check_for_double_names(iterable: tuple) -> None:
    """Raise error if any name isn't unique."""
    names = tuple(item.name for item in iterable)
    try:
        assert len(names) == len(set(names))
    except AssertionError:
        msg = "Each name of every object has to be unique! "
        msg = "Found double name in {}!".format(names)
        raise ValueError(msg)


def _not_attached_yet(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AttributeError:
            msg = "Perhaps property hasn't been defined yet because TrackMaker hasn't "
            msg += "been attached to a SegmentMaker yet. Return None!"
            logging.warn(msg)
            return None

    return wrapper


class _NamedObject(object):
    """General mother class for named objects."""

    def __init__(self, name: str) -> None:
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name


class MetaTrack(_NamedObject):
    """MetaTrack objects represent one voice within a complete composition.

    Volume and panning arguments are for stereo mixdown.
    panning: 0 means completely left and 1 means completely right.
    """

    def __init__(
        self, name: str, n_staves: int = 1, volume: float = 1, panning: float = 0.5
    ):
        _NamedObject.__init__(self, name)
        self._volume = volume
        self._n_staves = n_staves
        self._volume_left, self._volume_right = self._get_panning_arguments(panning)

    def __repr__(self) -> str:
        return "MetaTrack({})".format(self.name)

    @staticmethod
    def _get_panning_arguments(pan) -> tuple:
        return 1 - pan, pan

    @property
    def n_staves(self) -> float:
        return self._n_staves

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def volume_left(self) -> float:
        return self._volume_left

    @property
    def volume_right(self) -> float:
        return self._volume_right


class Orchestration(object):
    def __init__(self, *meta_tracks) -> None:
        _check_for_double_names(meta_tracks)
        self.__tracks = {mt.name: mt for mt in meta_tracks}

    def __repr__(self) -> str:
        return "Orchestration({})".format(self.__tracks)

    def __iter__(self) -> tuple:
        return iter(self.__tracks)

    def __getitem__(self, key: str) -> MetaTrack:
        return self.__tracks[key]

    def __len__(self) -> int:
        return len(self.__tracks)


class MusObject(object):
    def __init__(self, resolution: int = None):
        if resolution is None:
            resolution = STANDARD_RESOLUTION

        self._tmp_name = ".track_{}".format(uuid.uuid4().hex)
        self.resolution = resolution

    def _make_lilypond_file(self) -> abjad.LilyPondFile:
        layout_block = abjad.Block("layout")
        layout_block.items.append(r"indent = 0\mm")
        layout_block.items.append(r"line-width = 550\mm")
        layout_block.items.append(r"ragged-last = ##f")
        layout_block.items.append(r"ragged-right = ##f")

        score_block = abjad.Block("score")
        score_block.items.append(self.score)

        return abjad.LilyPondFile(
            lilypond_version_token=abjad.LilyPondVersionToken("2.19.83"),
            global_staff_size=22.45,
            includes=["lilypond-book-preamble.ly"],
            items=[layout_block, score_block],
        )

    @abc.abstractproperty
    def sound_engine(self) -> synthesis.SoundEngine:
        raise NotImplementedError

    @abc.abstractproperty
    def score(self) -> abjad.Score:
        raise NotImplementedError

    def notate(self, name: str) -> subprocess.Popen:
        lf = self._make_lilypond_file()
        lily_name = "{}.ly".format(name)

        with open(lily_name, "w") as f:
            f.write(lily.EKMELILY_PREAMBLE)
            f.write(format(lf))

        return subprocess.Popen(
            [
                "lilypond",
                "--png",
                "-dresolution={}".format(self.resolution),
                "-o{}".format(name),
                lily_name,
            ]
        )

    def synthesize(self, name: str) -> subprocess.Popen:
        return self.sound_engine.render(name)

    def show(self) -> None:
        self.notate(self._tmp_name).wait()
        subprocess.Popen(["xdg-open", "{}.png".format(self._tmp_name)]).wait()
        time.sleep(3)
        for ending in (
            ".ly",
            "-1.eps",
            "-systems.tex",
            "-systems.texi",
            "-systems.count",
            ".png",
        ):
            os.remove("{}{}".format(self._tmp_name, ending))

    def play(self) -> None:
        process = self.synthesize(self._tmp_name)
        process.wait()
        subprocess.call(["xdg-open", "{}.wav".format(self._tmp_name)])
        os.remove("{}.wav".format(self._tmp_name))


class Segment(MusObject):
    def __init__(self, orchestration: Orchestration, resolution: int = None, **tracks):
        for track_name in orchestration:

            try:
                assert track_name in tracks

            except AssertionError:
                msg = "Found undefined track {}. ".format(track_name)
                msg += "All tracks that has been definied in Orchestration have to be "
                msg += "passed to the Segment object."
                raise ValueError(msg)

        for track_name in tracks:
            setattr(self, track_name, tracks[track_name])

        self._orchestration = orchestration
        super().__init__(resolution=resolution)

    @property
    def score(self) -> abjad.Score:
        sco = abjad.Score(
            [
                abjad.mutate(getattr(self, track_name).abjad).copy()
                for track_name in self.orchestration
            ]
        )
        return sco

    @property
    def orchestration(self) -> Orchestration:
        return self._orchestration

    @property
    def sound_engine(self) -> synthesis.SoundEngine:
        return None

    def synthesize(self, name: str = None) -> subprocess.Popen:
        # anders als bei notation müssen bei synthese hier die subpfade der klangdateien
        # der einzelnen tracks bekannt sein. am besten würdest du die klangdateien
        # wahrscheinlich in einem bestimmten ordner abspeichern, der heißen würde:

        # build/chapter-name/segment-name/soundfiles/tracks/track-name.wav
        # es gäbe noch .../soundfiles/segment-name.wav

        # build/chapter-name/segment-name

        # res_sf_names = None
        for process in tuple(self.track.synthesize()):
            pass


class SegmentMaker(abc.ABC):
    _segment_class = Segment

    def __init__(self) -> None:
        for meta_track in self.orchestration:
            self.attach(**{meta_track: EmptyTrackMaker()})

    def __call__(self) -> Segment:
        tracks = {
            meta_track: getattr(self, meta_track)() for meta_track in self.orchestration
        }
        return self._segment_class(self.orchestration, **tracks)

    @property
    def duration(self) -> abjad.Duration:
        """Return notated duration of resulting Segment."""
        return sum(b.duration for b in self.bars)

    @abc.abstractproperty
    def bars(self) -> tuple:
        """Return abjad.TimeSignature object for each bar."""
        raise NotImplementedError

    @abc.abstractproperty
    def ratio2pitchclass_dict(self) -> dict:
        """Return entry for each appearing ratio and its abjad.PitchClass equivalent."""
        raise NotImplementedError

    @abc.abstractproperty
    def musdat(self) -> dict:
        """Return musical data that this SegmentMaker knows."""
        raise NotImplementedError

    @abc.abstractproperty
    def orchestration(self) -> Orchestration:
        """Return Orchestration"""
        raise NotImplementedError

    def attach(self, **kwargs) -> None:
        for track_name in kwargs:

            try:
                meta_track = self.orchestration[track_name]
            except KeyError:
                msg = "No MetaTack with name {} has had been defined in ".format(
                    track_name
                )
                msg += "the Orchestration."
                raise NotImplementedError(msg)

            track_maker = kwargs[track_name]
            track_maker.attach(self, meta_track)
            setattr(self, track_name, track_maker)


class Track(MusObject):
    def __init__(
        self,
        abjad_data: abjad.StaffGroup,
        sound_engine: synthesis.SoundEngine,
        resolution: int = None,
    ):
        self._abjad = abjad_data
        self._se = sound_engine
        super().__init__(resolution=resolution)

    @property
    def abjad(self) -> abjad.StaffGroup:
        return self._abjad

    @property
    def score(self):
        return abjad.Score([abjad.mutate(self.abjad).copy()])

    @property
    def sound_engine(self) -> synthesis.SoundEngine:
        return self._se


class TrackMaker(abc.ABC):
    _track_class = Track

    _ts2grid_size = {
        abjad.TimeSignature((2, 4)): fractions.Fraction(1, 4),
        abjad.TimeSignature((4, 4)): fractions.Fraction(1, 4),
        abjad.TimeSignature((3, 4)): fractions.Fraction(1, 4),
        abjad.TimeSignature((6, 8)): fractions.Fraction(3, 8),
        abjad.TimeSignature((5, 4)): fractions.Fraction(1, 4),
        abjad.TimeSignature((10, 8)): fractions.Fraction(5, 8),
    }

    @abc.abstractmethod
    def make_musdat(
        self, segment_maker: SegmentMaker, meta_track: MetaTrack
    ) -> old.PolyLine:
        raise NotImplementedError

    @abc.abstractmethod
    def make_sound_engine(self) -> synthesis.SoundEngine:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> lily.NOventLine:
        try:
            return self.musdat[idx]
        except TypeError:
            return None

    def __call__(self) -> Track:
        # 1. make abjad data
        staves = []
        for line in self.musdat:
            staves.append(
                self._convert_novent_line2abjad_staff(
                    line, self.bars, self.ratio2pitchclass_dict
                )
            )

        abjad_data = abjad.Container(staves)
        abjad_data.simultaneous = True

        # 2. generate sound engine
        sound_engine = self.make_sound_engine()

        return self._track_class(abjad_data, sound_engine)

    def attach(self, segment_maker: SegmentMaker, meta_track: MetaTrack) -> None:
        self._bars = segment_maker.bars
        self._ratio2pitchclass_dict = segment_maker.ratio2pitchclass_dict
        self._musdat = self.make_musdat(segment_maker, meta_track)

        try:
            assert len(self._musdat) == meta_track.n_staves
        except AssertionError:
            msg = "Resulting musdat only has {} NOvent - lines while {} ".format(
                len(self._musdat), meta_track.n_staves
            )
            msg += "has had been expected from the orchestra definition."
            raise ValueError(msg)

    @property
    @_not_attached_yet
    def musdat(self) -> old.PolyLine:
        return self._musdat

    @property
    @_not_attached_yet
    def bars(self) -> tuple:
        return self._bars

    @property
    @_not_attached_yet
    def ratio2pitchclass_dict(self) -> dict:
        return self._ratio2pitchclass_dict

    # GENERAL NOTATION / ABJAD - CONVERSION - METHODS #

    @staticmethod
    def _convert_symbolic_novent_line2asterisked_novent_line(
        novent_line: lily.NOventLine
    ) -> lily.NOventLine:
        new_line = []

        tempo = abjad.MetronomeMark((1, 4), 60)
        for novent in novent_line:
            novent_copied = novent.copy()

            if novent_copied.tempo:
                tempo = novent_copied.tempo.abjad

            novent_copied.delay, novent_copied.duration = (
                tempo.duration_to_milliseconds(d) / 1000
                for d in (novent.delay, novent.duration)
            )
            new_line.append(novent_copied)

        return lily.NOventLine(new_line)

    @classmethod
    def _mk_grids(cls, bars: tuple) -> tuple:
        bar_grid = tuple(
            fractions.Fraction(ts.numerator, ts.denominator) for ts in bars
        )
        grid = functools.reduce(
            operator.add,
            tuple(
                cls._make_metrical_grid_according_to_time_signature(ts) for ts in bars
            ),
        )
        # for ts whose numerator != 2**x an additional cautious grid is necessary!
        cautious_grid = []
        for ts in bars:
            gs = cls._ts2grid_size[ts]
            if gs.numerator in (3, 5, 7):
                size = ts.duration
                cautious_gs = fractions.Fraction(1, 8)
                for _ in range(size // cautious_gs):
                    cautious_grid.append(cautious_gs)
            else:
                cautious_grid.append(fractions.Fraction(ts.duration))

        return bar_grid, cautious_grid, grid

    @staticmethod
    def _seperate_tone_with_glissando(self) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _convert_mu_pitch2named_pitch(
        pitch: ji.JIPitch, ratio2pitchclass_dict: dict = None
    ) -> abjad.NamedPitch:
        if pitch.is_empty:
            np = None
        else:
            if ratio2pitchclass_dict:
                np = lily.convert2abjad_pitch(pitch, ratio2pitchclass_dict)

            else:
                np = abjad.NamedPitch(lily.round_cents_to_12th_tone(pitch.cents))

        return np

    @staticmethod
    def _seperate_by_grid(
        start: fractions.Fraction,
        stop: fractions.Fraction,
        grid: tuple,
        hard_cut: bool = False,
    ) -> tuple:
        def detect_data(i: int, group: int) -> tuple:
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

            return is_connectable, new_delay

        absolute_grid = tools.accumulate_from_zero(grid)
        grid_start = bisect.bisect_right(absolute_grid, start) - 1
        grid_stop = bisect.bisect_right(absolute_grid, stop)
        passed_groups = tuple(range(grid_start, grid_stop, 1))

        if len(passed_groups) == 1:
            return (stop - start,)

        else:
            if hard_cut:
                # TODO(make this less ugly)

                for position, item in enumerate(absolute_grid):
                    if item > start:
                        break
                    else:
                        start_position = position

                positions = [start] + list(
                    absolute_grid[
                        start_position + 1 : start_position + len(passed_groups) + 1
                    ]
                )
                if stop == positions[-2]:
                    positions = positions[:-1]
                else:
                    positions[-1] = stop
                return tuple(b - a for a, b in zip(positions, positions[1:]))

            else:
                delays = []
                is_connectable_per_delay = []
                for i, group in enumerate(passed_groups):
                    is_connectable, new_delay = detect_data(i, group)
                    if new_delay > 0:
                        delays.append(new_delay)
                        is_connectable_per_delay.append(is_connectable)

                length_delays = len(delays)
                if length_delays == 1:
                    return tuple(delays)

                connectable_range = [
                    int(not is_connectable_per_delay[0]),
                    length_delays,
                ]

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
    def _seperate_by_assignability(
        duration: fractions.Fraction,
        max_duration: fractions.Fraction = fractions.Fraction(1, 1),
    ) -> tuple:
        def find_sum_in_numbers(numbers, solution) -> tuple:
            result = []
            # from smallest biggest to smallest
            nums = reversed(sorted(numbers))
            current_num = next(nums)
            while sum(result) != solution:
                if sum(result) + current_num <= solution:
                    result.append(current_num)
                else:
                    current_num = next(nums)
            return tuple(result)

        # easy claim for standard note duration
        if abjad.Duration(duration).is_assignable and duration <= max_duration:
            return (abjad.Duration(duration),)

        # top and bottom
        numerator = duration.numerator
        denominator = duration.denominator

        # we only need note durations > 1 / denominator
        possible_durations = [
            fractions.Fraction(i, denominator)
            for i in range(1, numerator + 1)
            # only standard note durations
            if abjad.Duration(i, denominator).is_assignable
            and (i / denominator) <= max_duration
        ]

        # find the right combination
        solution = find_sum_in_numbers(possible_durations, duration)
        return solution

    @classmethod
    def _make_metrical_grid_according_to_time_signature(
        cls, time_signature: abjad.TimeSignature
    ) -> tuple:
        try:
            gs = cls._ts2grid_size[time_signature]
        except KeyError:
            msg = "Gridsize for TimeSignature {} hasn't been definied yet.".format(
                time_signature
            )
            raise NotImplementedError(msg)

        return tuple(gs for i in range(float(time_signature.duration) // gs))

    @classmethod
    def _make_absolute_grid_for_time_signature_depending_beaming(
        cls, time_signatures: tuple
    ) -> tuple:
        return tools.accumulate_from_zero(
            functools.reduce(
                operator.add,
                tuple(
                    cls._make_metrical_grid_according_to_time_signature(ts)
                    for ts in time_signatures
                ),
            )
        )

    @staticmethod
    def _apply_beams(notes: abjad.Voice, absolute_grid: tuple = None) -> None:
        durations = tuple(float(n.written_duration) for n in notes)
        absolute_durations = tools.accumulate_from_zero(durations)

        if absolute_grid is None:
            absolute_grid = tools.accumulate_from_zero(
                tuple(
                    fractions.Fraction(1, 4)
                    for i in range(sum(durations) // fractions.Fraction(1, 4))
                )
            )

        duration_positions = []
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
                idx1 = len(notes)

            add_beams = False
            for n in notes[idx0:idx1]:
                if type(n) not in (abjad.Rest, abjad.MultimeasureRest):
                    add_beams = True
                    break

            if idx1 - idx0 < 2:
                add_beams = False

            if add_beams is True:
                abjad.attach(abjad.StartBeam(), notes[idx0])
                abjad.attach(abjad.StopBeam(), notes[idx1 - 1])

    @staticmethod
    def _subdivide_by_measures(chords: list, time_signatures: tuple) -> list:
        time_signatures = iter(time_signatures)

        bars = []

        last_ts = None

        container = abjad.Container([])
        current_ts = next(time_signatures)
        current_size = 0

        for chord in chords:
            if current_size >= current_ts.duration:

                if last_ts != current_ts:
                    abjad.attach(current_ts, container[0])

                bars.append(container)

                container = abjad.Container([chord])
                last_ts = current_ts
                current_ts = next(time_signatures)
                current_size = chord.written_duration

            else:
                container.append(chord)
                current_size += chord.written_duration

        if container:
            bars.append(container)
            if current_size >= current_ts.duration:
                if last_ts != current_ts:
                    abjad.attach(current_ts, container[0])

        return bars

    @staticmethod
    def _attach_on_off_attachments(notes: list, on_off_attachments: dict) -> None:
        # attach on-off attachments
        for group in on_off_attachments:
            previous_attachment = None
            for OFdata in on_off_attachments[group]:
                attachment, novent, subnotes_positions = OFdata
                tests = (
                    attachment != previous_attachment,
                    previous_attachment != attachments.Hauptstimme(True, True),
                )
                if all(tests):

                    special_hauptstimme_case_tests = (
                        attachment == attachments.Hauptstimme(True, True),
                        subnotes_positions[-1] - subnotes_positions[0] > 1,
                    )

                    if all(special_hauptstimme_case_tests):
                        attachments.Hauptstimme(True).attach(
                            notes[subnotes_positions[0]], novent
                        )
                        attachments.Hauptstimme(False).attach(
                            notes[subnotes_positions[-1] - 1], novent
                        )

                    else:

                        # special case for Hauptstimme - Attachment
                        if attachment == attachments.Hauptstimme(False):
                            idx = subnotes_positions[-1] - 1
                        else:
                            idx = subnotes_positions[0]

                        attachment.attach(notes[idx], novent)
                previous_attachment = attachment

    @staticmethod
    def _divide2subdelays(
        novent: lily.NOvent, cautious_grid: tuple, grid: tuple, bar_grid: tuple
    ) -> None:
        start, stop = (
            fractions.Fraction(novent.delay),
            fractions.Fraction(novent.duration),
        )
        seperated_by_bar = tools.accumulate_from_n(
            TrackMaker._seperate_by_grid(
                fractions.Fraction(start),
                fractions.Fraction(stop),
                bar_grid,
                hard_cut=True,
            ),
            start,
        )
        sub_delays = tools.accumulate_from_n(
            functools.reduce(
                operator.add,
                tuple(
                    TrackMaker._seperate_by_grid(sta, sto, grid)
                    for sta, sto in zip(seperated_by_bar, seperated_by_bar[1:])
                ),
            ),
            start,
        )
        sub_delays = functools.reduce(
            operator.add,
            tuple(
                functools.reduce(
                    operator.add,
                    tuple(
                        TrackMaker._seperate_by_assignability(d)
                        for d in TrackMaker._seperate_by_grid(sta, sto, cautious_grid)
                    ),
                )
                for sta, sto in zip(sub_delays, sub_delays[1:])
            ),
        )
        return sub_delays

    @staticmethod
    def _make_subnotes(
        abjad_pitches: tuple,
        subdelays: tuple,
        previous_duration: float,
        absolute_bar_grid: tuple,
    ) -> tuple:
        subnotes = []
        absolute_subdelays = tools.accumulate_from_n(subdelays, previous_duration)

        for delay, start, stop in zip(
            subdelays, absolute_subdelays, absolute_subdelays[1:]
        ):
            if abjad_pitches is None:
                if start in absolute_bar_grid and stop in absolute_bar_grid:
                    obj = abjad.MultimeasureRest(delay)

                else:
                    obj = abjad.Rest(delay)

            else:
                obj = abjad.Chord(abjad_pitches, delay)

            subnotes.append(obj)

        return tuple(subnotes)

    @classmethod
    def _convert_novent_line2abjad_staff(
        cls,
        novent_line: lily.NOventLine,
        time_signatures: tuple,
        ratio2pitchclass_dict: dict = None,
    ) -> abjad.Staff:

        bar_grid, cautious_grid, grid = cls._mk_grids(time_signatures)
        absolute_bar_grid = tools.accumulate_from_zero(bar_grid)

        notes = []

        on_off_attachments = {}
        previous_duration = 0

        for novent in novent_line.convert2absolute():
            # when adding glissando, the tone has to be seperated first and then each
            # seperated part has to get seperated again by bar lines and grid lines

            if novent.pitch:
                abjad_pitches = tuple(
                    TrackMaker._convert_mu_pitch2named_pitch(p, ratio2pitchclass_dict)
                    for p in novent.pitch
                )

            else:
                abjad_pitches = None

            subdelays = TrackMaker._divide2subdelays(
                novent, cautious_grid, grid, bar_grid
            )
            subnotes = TrackMaker._make_subnotes(
                abjad_pitches, subdelays, previous_duration, absolute_bar_grid
            )
            previous_duration += sum(subdelays)

            for attachment in filter(lambda at: bool(at), novent.attachments):

                # save on-off attachments and attach them later
                if attachment.is_on_off_notation:
                    if attachment.name not in on_off_attachments:
                        on_off_attachments.update({attachment.name: []})

                    length_notes = len(notes)
                    OFdata = (
                        attachment,
                        novent,
                        (length_notes, length_notes + len(subnotes)),
                    )

                    on_off_attachments[attachment.name].append(OFdata)

                # attach isolated attachments
                else:
                    if attachment.attach_on_each_part:
                        for note in subnotes:
                            attachment.attach(note, novent)
                    else:
                        attachment.attach(subnotes[0], novent)

            # tie notes
            if abjad_pitches is not None and len(subnotes) > 1:
                for note in subnotes[:-1]:
                    abjad.attach(abjad.Tie(), note)

            notes.extend(subnotes)

        TrackMaker._apply_beams(
            notes,
            TrackMaker._make_absolute_grid_for_time_signature_depending_beaming(
                time_signatures
            ),
        )
        TrackMaker._attach_on_off_attachments(notes, on_off_attachments)

        staff = abjad.Staff(TrackMaker._subdivide_by_measures(notes, time_signatures))
        abjad.setting(staff).auto_beaming = False
        abjad.attach(abjad.LilyPondLiteral("\\numericTimeSignature"), staff[0][0])

        return staff


class EmptyTrackMaker(TrackMaker):
    """Only return empty sound file and empty staves per staff."""

    def make_musdat(
        self, segment_maker: SegmentMaker, meta_track: MetaTrack
    ) -> old.PolyLine:
        pl = []
        dur = segment_maker.duration
        for staff in range(meta_track.n_staves):
            pl.append(lily.NOventLine([lily.NOvent(duration=dur, delay=dur)]))

        return old.PolyLine(pl)

    def make_sound_engine(self) -> synthesis.SoundEngine:
        return synthesis.SilenceEngine(1)
