"""MU aims helping to organise acousmatic compositions.

One MU.MU object shall represent one piece. It is composed of different
Segment objects.

For the usual workflow it is expected that the user...
    (1) first defines an Orchestration class for the complete piece.
    (2) then defines classes that inherit from the MU.Segment class.
    (3) last but not least, initalise objects from those Segment classes
        that are the input for the MU object.

Then single tracks can be rendered through MUs render method.
For a better composing workflow, a stereo mixdown can be created
with MUs 'stereo_mixdown' method.
"""

# TODO(Add MU objects for score generation [similar organisiation of segments that get
# glued to bigger structures])

import functools
import operator
import os
import subprocess
import progressbar

from mu.utils import interpolations
from mu.utils import tools

from mutools import csound
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


class Track(_NamedObject):
    """Track objects represent one voice within a complete composition.

    Volume and panning arguments are for stereo mixdown.
    panning: 0 means completely left and 1 means completely right.
    """

    def __init__(self, name: str, volume: float = 1, panning: float = 0.5):
        _NamedObject.__init__(self, name)
        self._volume = volume
        self._volume_left, self._volume_right = self._get_panning_arguments(panning)

    def __repr__(self) -> str:
        return "Track({})".format(self.name)

    @staticmethod
    def _get_panning_arguments(pan) -> tuple:
        return 1 - pan, pan

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
    def __init__(self, *tracks) -> None:
        _check_for_double_names(tracks)
        self.__tracks = tracks

    def __repr__(self) -> str:
        return "Orchestration({})".format(self.__tracks)

    def __iter__(self) -> tuple:
        return iter(self.__tracks)

    def __getitem__(self, idx) -> Track:
        return self.__tracks[idx]

    def __len__(self) -> int:
        return len(self.__tracks)


class _AttachEnvelope(synthesis.BasedCsoundEngine):
    def __init__(
        self, path: str, envelopes: tuple, upper_process: subprocess.Popen
    ) -> None:
        self.__path = "{}.wav".format(path)
        self.__changed_path = "{}_copied.wav".format(path)
        self.__envelopes = envelopes
        self.__upper_process = upper_process

    @property
    def cname(self) -> str:
        return ".enveloper"

    @property
    def orc(self) -> str:
        envelope_names = []
        envelope_lines = []
        for idx, interpolation in enumerate(self.__envelopes):
            name = "kEnvelope{}".format(idx)
            linseg = ", ".join(
                str(item)
                for item in functools.reduce(
                    operator.add,
                    tuple(
                        (point.value, point.delay) if point.delay else (point.value,)
                        for point in interpolation
                    ),
                )
            )
            envelope = "{} linseg {}".format(name, linseg)
            envelope_names.append(name)
            envelope_lines.append(envelope)

        lines = (r"0dbfs=1", r"instr 1")
        lines += tuple(envelope_lines)
        lines += (
            r'aSig diskin2 "{}", 1, 0, 0, 6, 4'.format(self.__changed_path),
            r"out {}".format(" * ".join(["aSig"] + envelope_names)),
            r"endin",
        )
        return "\n".join(lines)

    @property
    def sco(self) -> str:
        return "i1 0 {}".format(self.__duration)

    def render(self, name: str) -> subprocess.Popen:
        if self.__upper_process is not None:
            self.__upper_process.wait()
        subprocess.call(["mv", self.__path, self.__changed_path])
        self.__duration = synthesis.pyo.sndinfo(self.__changed_path)[1]
        return super().render(name)


class _SegmentMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        for track in x.orchestration:
            setattr(x, track.name, {"start": 0, "duration": 0, "sound_engine": None})
        return x


class Segment(_NamedObject, metaclass=_SegmentMeta):
    orchestration = Orchestration()

    def __init__(
        self,
        name: str,
        start: float = 0,
        tracks2ignore: tuple = tuple([]),
        volume_envelope: interpolations.InterpolationLine = None,
        volume_envelope_per_track: dict = dict([]),
        **kwargs
    ) -> None:
        _NamedObject.__init__(self, name)
        self._volume_envelope = volume_envelope
        self._volume_envelope_per_track = volume_envelope_per_track
        self.__data = kwargs
        self.__start = start
        # tracks that can be ignored for calculating the duration of a Segment
        self.__tracks2ignore = tracks2ignore

        for name in kwargs:

            try:
                assert hasattr(self, name)
            except AssertionError:
                msg = "No track named {} known.".format(name)
                raise AttributeError(msg)

            setattr(self, name, kwargs[name])

    @property
    def start(self) -> float:
        return self.__start

    def __repr__(self) -> str:
        return "Segment({})".format(self.name)

    def render(self, path: str) -> tuple:
        """Render all sound files.

        Make empty sound file for any track that hasn't been initalized.
        """
        processes = []
        duration = self.duration
        for track in self.orchestration:
            data = getattr(self, track.name)
            sound_engine = data["sound_engine"]
            is_silent = False

            if sound_engine is None:
                sound_engine = synthesis.SilenceEngine(duration)
                is_silent = True

            if isinstance(sound_engine, synthesis.PyoEngine):
                sound_engine = sound_engine.copy()

            local_path = "{}/{}".format(path, track.name)
            process = sound_engine.render(local_path)

            if not is_silent:
                try:
                    track_envelope = self._volume_envelope_per_track[track.name]
                except KeyError:
                    track_envelope = None

                envelopes = tuple(
                    env for env in (self._volume_envelope, track_envelope) if env
                )

                if envelopes:
                    additional_process = _AttachEnvelope(
                        local_path, envelopes, process
                    ).render(local_path)
                    processes.append(additional_process)

            processes.append(process)

        return tuple(processes)

    @property
    def duration(self) -> float:
        return max(
            getattr(self, track.name)["start"] + getattr(self, track.name)["duration"]
            for track in self.orchestration
            if track.name not in self.__tracks2ignore
        )


class MU(_NamedObject):
    _concatenated_path = "concatenated"

    def __init__(
        self, name: str, orchestration: Orchestration, *segment, tail: float = 10
    ):
        _check_for_double_names(segment)
        _NamedObject.__init__(self, name)
        self.tail = tail
        self._segments = segment
        self._segments_by_name = {seg.name: seg for seg in segment}
        self._orchestration = orchestration

        self.mkdir(self.name)
        self.mkdir("{}/{}".format(self.name, self._concatenated_path))

        for segment in self.segments:
            self.mkdir("{}/{}".format(self.name, segment.name))

    def __repr__(self) -> str:
        return "MU!({})".format(self.name)

    @staticmethod
    def mkdir(path: str) -> None:
        """mkdir that ignores FileExistsError."""
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @property
    def orchestration(self) -> Orchestration:
        return self._orchestration

    @property
    def segments(self) -> tuple:
        return self._segments

    def get_segment_by_name(self, name: str) -> tuple:
        return self._segments_by_name[name]

    @staticmethod
    def _make_sampler_orc(n_channels: int = 1) -> str:
        lines = (
            r"0dbfs=1",
            r"nchnls={}".format(n_channels),
            r"instr 1",
            r"asig diskin2 p4, 1, 0, 0, 6, 4",
            r"asig = asig * p5",
        )

        if n_channels == 1:
            lines += (r"out asig",)
        elif n_channels == 2:
            lines += (r"outs asig * p6, asig * p7",)
        else:
            raise NotImplementedError

        lines += (r"endin",)

        return " \n".join(lines)

    def concatenate(self) -> None:
        start_positions_of_tracks_for_first_segment = tuple(
            getattr(self.segments[0], track.name)["start"]
            for track in self.orchestration
        )
        minima_start_position_of_tracks_for_first_segment = min(
            start_positions_of_tracks_for_first_segment
        )
        added_value_for_start_position_for_first_segment = abs(
            minima_start_position_of_tracks_for_first_segment
        )

        adapted_duration_per_segment = list(
            segment.duration for segment in self.segments
        )
        for idx, segment in enumerate(self.segments[1:]):
            adapted_duration_per_segment[idx] += segment.start

        adapted_duration_per_segment[
            0
        ] += added_value_for_start_position_for_first_segment
        start_position_per_segment = tuple(
            position + self.segments[0].start
            for position in tools.accumulate_from_zero(adapted_duration_per_segment)
        )

        for start_position in start_position_per_segment:
            try:
                assert start_position >= 0
            except AssertionError:
                msg = "Segment has a too low start value."
                raise ValueError(msg)

        orc_name = ".concatenate"
        sco_name = ".concatenate"

        processes = []

        print("CONCATENATING TRACKS")

        with progressbar.ProgressBar(max_value=len(self.orchestration)) as bar:
            for track_idx, track in enumerate(self.orchestration):

                local_orc_name = "{}_{}.orc".format(orc_name, track_idx)
                local_sco_name = "{}_{}.sco".format(sco_name, track_idx)

                with open(local_orc_name, "w") as f:
                    f.write(self._make_sampler_orc(n_channels=1))

                relevant_data = []  # start, duration, path

                is_first_segment = True

                for start_position_of_segment, segment in zip(
                    start_position_per_segment, self.segments
                ):
                    path = "{}/{}/{}.wav".format(self.name, segment.name, track.name)
                    start_position = (
                        start_position_of_segment
                        + getattr(segment, track.name)["start"]
                    )

                    if is_first_segment:
                        start_position += (
                            added_value_for_start_position_for_first_segment
                        )

                    duration = getattr(segment, track.name)["duration"]
                    if duration < segment.duration:
                        duration = segment.duration

                    duration += self.tail
                    relevant_data.append((start_position, duration, path))

                    is_first_segment = False

                sco = " \n".join(
                    tuple('i1 {} {} "{}" 1'.format(*d) for d in relevant_data)
                )

                with open(local_sco_name, "w") as f:
                    f.write(sco)

                sf_name = "{}/{}/{}.wav".format(
                    self.name, self._concatenated_path, track.name
                )
                processes.append(
                    csound.render_csound(sf_name, local_orc_name, local_sco_name)
                )

                bar.update(track_idx)

        for process in processes:
            process.wait()

    def render(self) -> None:
        processes = []
        n_segments = len(self.segments)

        print("RENDER {} SEGMENTS.".format(n_segments))

        with progressbar.ProgressBar(max_value=n_segments) as bar:
            for idx, segment in enumerate(self.segments):
                processes.extend(self.render_segment(segment.name))
                bar.update(idx)

        print("WAITING FOR ALL SUBPROCESSES TO BE FINISHED.")
        with progressbar.ProgressBar(max_value=len(processes)) as bar:
            for idx, process in enumerate(processes):
                if isinstance(process, subprocess.Popen):
                    process.wait()
                bar.update(idx)

        self.concatenate()

    def render_segment(self, segment_name: str) -> None:
        return self._segments_by_name[segment_name].render(
            "{}/{}".format(self.name, segment_name)
        )

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def stereo_mixdown(self, do_render: bool = True) -> None:
        if do_render:
            self.render()

        print("START STEREO MIXDOWN")
        sf_name = "{}/stereo_mixdown.wav".format(self.name)
        orc_name = ".mixdown.orc"
        sco_name = ".mixdown.sco"

        with open(orc_name, "w") as f:
            f.write(self._make_sampler_orc(n_channels=2))

        path_per_concatenated_file = tuple(
            "{}/{}/{}.wav".format(self.name, self._concatenated_path, track.name)
            for track in self.orchestration
        )

        sco = " \n".join(
            tuple(
                'i1 0 {} "{}" {} {} {}'.format(
                    synthesis.pyo.sndinfo(track_path)[1] + self.tail,
                    track_path,
                    track.volume,
                    track.volume_left,
                    track.volume_right,
                )
                for track_path, track in zip(
                    path_per_concatenated_file, self.orchestration
                )
            )
        )

        with open(sco_name, "w") as f:
            f.write(sco)

        csound.render_csound(sf_name, orc_name, sco_name).wait()
