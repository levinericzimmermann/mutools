import abc
import fractions
import functools
import math
import operator
import subprocess
import pathlib
import uuid

import abjad

from mu.mel import ji
from mu.utils import tools
from mutools import lily
from mutools import synthesis


def _check_for_double_names(iterable: tuple) -> None:
    """Raise error if any name isn't unique."""
    names = tuple(item.name for item in iterable)
    try:
        assert len(names) == len(set(names))
    except AssertionError:
        msg = "Each name of every object has to be unique! "
        msg = "Found double name in {}!".format(names)
        raise ValueError(msg)


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
        return "Track({})".format(self.name)

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
        self.__tracks = meta_tracks

    def __repr__(self) -> str:
        return "Orchestration({})".format(self.__tracks)

    def __iter__(self) -> tuple:
        return iter(self.__tracks)

    def __getitem__(self, idx) -> MetaTrack:
        return self.__tracks[idx]

    def __len__(self) -> int:
        return len(self.__tracks)


class Segment(object):
    pass


class _MetaSegmentMaker(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        for meta_track in x.orchestration:
            setattr(x, meta_track.name, TheEmptyTrackMaker)
        return x


class SegmentMaker(abc.ABC):
    def __init__(self, bars: tuple) -> None:
        pass

    def __call__(self) -> Segment:
        pass

    @abc.abstractproperty
    def bars(self) -> tuple:
        """Return abjad.TimeSignature object for each bar."""
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
        pass


class Track(object):
    _tmp_name = "{}/../../tmp/track_{}".format(pathlib.Path.home(), uuid.uuid4().hex)

    def __init__(
        self, abjad_data: abjad.StaffGroup, sound_engine: synthesis.SoundEngine
    ):
        self._abjad = abjad_data
        self._se = sound_engine

    @property
    def abjad(self) -> abjad.StaffGroup:
        return self._abjad

    @property
    def sound_engine(self) -> synthesis.SoundEngine:
        return self._se

    def notate(self, name: str, convert2png: bool = True) -> None:
        score = abjad.Score([self.abjad])

        lf = abjad.LilyPondFile(
            score,
            lilypond_version_token=abjad.LilyPondVersionToken("2.19.83"),
            includes=["lilypond-book-preamble.ly"],
        )

        lily_name = "{}.ly".format(name)

        with open(lily_name, "w") as f:
            f.write(lily.EKMELILY_PREAMBLE)
            f.write(format(lf))

        if convert2png:
            subprocess.call(["lilypond", "--png", "-dresolution=400", lily_name])
        else:
            subprocess.call(["lilypond", lily_name])

    def synthesize(self, name: str) -> subprocess.Popen:
        return self.sound_engine(name).render()

    def show(self, reference_pitch: int = 0) -> None:
        self.notate(self._tmp_name, reference_pitch)
        subprocess.call(["o", "{}.png".format(self._tmp_name)])

    def play(self) -> None:
        process = self.synthesize(self._tmp_name)
        process.wait()
        subprocess.call(["o", "{}.wav".format(self._tmp_name)])


class TrackMaker(object):
    def __call__(self) -> Track:
        pass

    def attach(self, segment_maker: SegmentMaker) -> None:
        pass

    @staticmethod
    def _mk_bar_grid_and_grid(bars: tuple) -> tuple:
        bar_grid = tuple(
            fractions.Fraction(ts.numerator, ts.denominator) for ts in bars
        )
        grid = tuple(
            fractions.Fraction(1, 4)
            for i in range(int(math.ceil(sum(bar_grid) / fractions.Fraction(1, 4))))
        )
        return bar_grid, grid

    @staticmethod
    def _seperate_tone_with_glissando(self) -> tuple:
        pass

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
    def _subdivide_by_measures(chords: list, time_signatures: tuple) -> list:
        time_signatures = iter(time_signatures)

        bars = []

        last_ts = None

        container = abjad.Container([])
        current_ts = next(time_signatures)
        current_size = 0

        for chord in chords:
            if current_size == current_ts.duration:
                if last_ts != current_ts:
                    abjad.attach(current_ts, container[0])
                bars.append(container)

                container = abjad.Container([])
                last_ts = current_ts
                current_ts = next(time_signatures)
                current_size = 0

            else:
                container.append(chord)
                current_size += chord.written_duration

        return bars

    @staticmethod
    def _attach_on_off_attachments(notes: list, on_off_attachments: dict) -> None:
        # attach on-off attachments
        for group in on_off_attachments:
            previous_attachment = None
            for OFdata in group:
                attachment, novent, subnotes_positons = OFdata
                if attachment != previous_attachment:
                    attachment.attach(notes[subnotes_positons[0]], novent)

    @staticmethod
    def _divide2subdelays(novent: lily.NOvent, grid: tuple, bar_grid: tuple) -> None:
        seperated_by_bar = tools.accumulate_from_n(
            lily.seperate_by_grid(
                novent.delay, novent.duration, bar_grid, hard_cut=True
            ),
            novent.delay,
        )
        sub_delays = functools.reduce(
            operator.add,
            tuple(
                functools.reduce(
                    operator.add,
                    tuple(
                        lily.seperate_by_assignability(d)
                        for d in lily.seperate_by_grid(
                            novent.delay, novent.duration, grid
                        )
                    ),
                )
                for novent.delay, novent.duration in zip(
                    seperated_by_bar, seperated_by_bar[1:]
                )
            ),
        )
        return sub_delays

    @staticmethod
    def convert_novent_line2abjad_staff(
        novent_line: lily.NOventLine,
        time_signatures: tuple,
        ratio2pitchclass_dict: dict = None,
    ) -> abjad.Staff:

        bar_grid, grid = TrackMaker._mk_bar_grid_and_grid(time_signatures)

        notes = []

        on_off_attachments = {}

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

            subnotes = []
            for delay in TrackMaker._divide2subdelays(novent, grid, bar_grid):
                if abjad_pitches is None:
                    obj = abjad.Rest(delay)

                else:
                    obj = abjad.Chord(abjad_pitches, delay)

                subnotes.append(obj)

            # TODO make sure artifical harmonics will get attached first (if there is any)
            for attachment in novent.attachments:

                # save on-off attachments
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

        TrackMaker._attach_on_off_attachments(notes, on_off_attachments)

        staff = abjad.Staff(TrackMaker._subdivide_by_measures(notes, time_signatures))

        return staff


class _EmptyTackMaker(TrackMaker):
    # only return empty sound file and empty staves per staff
    pass


TheEmptyTrackMaker = _EmptyTackMaker()
