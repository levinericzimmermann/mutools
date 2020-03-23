import os

from mu.utils import tools

from mutools import csound
from mutools import synthesis


"""MU aims helping to organise acousmatic compositions.

One MU.MU object shall represent one piece. It is composed of different
Segment objects.

For the usual workflow it is expected that the user...
    (1) first defines an Orchestration class for the complete piece.
    (2) then defines classes that inherit from the MU.Segment class. To overwrite
        the default render method is especially crucial for this process.
    (3) last but not least, build objects from those Segment classes
        that are the input for the MU object.

Then single tracks can be rendered through MUs render method.
For a better composing workflow, a stereo mixdown can be created
with MUs 'stereo_mixdown' method.
"""


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
        self.__volume = volume
        self.__volume_left, self.__volume_right = self.__get_panning_arguments(panning)

    def __repr__(self) -> str:
        return "Track({})".format(self.name)

    @staticmethod
    def __get_panning_arguments(pan) -> tuple:
        return 1 - pan, pan

    @property
    def volume(self) -> float:
        return self.__volume

    @property
    def volume_left(self) -> float:
        return self.__volume_left

    @property
    def volume_right(self) -> float:
        return self.__volume_right


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


class _SegmentMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        for track in x.orchestration:
            setattr(x, track.name, {"start": 0, "duration": 0, "sound_engine": None})
        return x


class Segment(_NamedObject, metaclass=_SegmentMeta):
    orchestration = Orchestration()

    def __init__(self, name: str, **kwargs) -> None:
        _NamedObject.__init__(self, name)
        self.__data = kwargs

        for name in kwargs:

            try:
                assert hasattr(self, name)
            except AssertionError:
                msg = "No track named {} known.".format(name)
                raise AttributeError(msg)

            setattr(self, name, kwargs[name])

    def __repr__(self) -> str:
        return "Segment({})".format(self.name)

    def render(self, path: str) -> None:
        """Make empty sound file for any track that hasn't been initalized.

        If you don't want to lose this functionality, don't forget
            super().render(path)
        as a last line for the render method of inherited classes.
        """
        duration = self.duration
        for track in self.orchestration:
            data = getattr(self, track.name)
            sound_engine = data["sound_engine"]
            if sound_engine is None:
                sound_engine = synthesis.SilenceEngine(duration)
            if isinstance(sound_engine, synthesis.PyoEngine):
                sound_engine = sound_engine.copy()
            sound_engine.render("{}/{}".format(path, track.name))

    @property
    def duration(self) -> float:
        return max(
            getattr(self, track.name)["start"] + getattr(self, track.name)["duration"]
            for track in self.orchestration
        )


class MU(_NamedObject):
    __concatenated_path = "concatenated"

    def __init__(
        self, name: str, orchestration: Orchestration, *segment, tail: float = 10
    ):
        _check_for_double_names(segment)
        _NamedObject.__init__(self, name)
        self.tail = tail
        self.__segments = segment
        self.__segments_by_name = {seg.name: seg for seg in segment}
        self.__orchestration = orchestration

        self.mkdir(self.name)
        self.mkdir("{}/{}".format(self.name, self.__concatenated_path))

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
        return self.__orchestration

    @property
    def segments(self) -> tuple:
        return self.__segments

    def get_segment_by_name(self, name: str) -> tuple:
        return self.__segments_by_name[name]

    @staticmethod
    def __make_sampler_orc(n_channels: int = 1) -> str:
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

        duration_per_segment = list(segment.duration for segment in self.segments)
        duration_per_segment[0] += added_value_for_start_position_for_first_segment
        start_position_per_segment = tools.accumulate_from_zero(duration_per_segment)

        orc_name = ".concatenate.orc"
        sco_name = ".concatenate.sco"

        with open(orc_name, "w") as f:
            f.write(self.__make_sampler_orc(n_channels=1))

        for track in self.orchestration:
            relevant_data = []  # start, duration, path

            is_first_segment = True

            for start_position_of_segment, segment in zip(
                start_position_per_segment, self.segments
            ):
                path = "{}/{}/{}.wav".format(self.name, segment.name, track.name)
                start_position = (
                    start_position_of_segment + getattr(segment, track.name)["start"]
                )

                if is_first_segment:
                    start_position += added_value_for_start_position_for_first_segment

                duration = segment.duration + self.tail
                relevant_data.append((start_position, duration, path))

                is_first_segment = False

            sco = " \n".join(tuple('i1 {} {} "{}" 1'.format(*d) for d in relevant_data))

            with open(sco_name, "w") as f:
                f.write(sco)

            sf_name = "{}/{}/{}.wav".format(
                self.name, self.__concatenated_path, track.name
            )
            csound.render_csound(sf_name, orc_name, sco_name)
            os.remove(sco_name)

        os.remove(orc_name)

    def render(self) -> None:
        for segment in self.segments:
            self.render_segment(segment.name)
        self.concatenate()

    def render_segment(self, segment_name: str) -> None:
        self.__segments_by_name[segment_name].render(
            "{}/{}".format(self.name, segment_name)
        )

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def stereo_mixdown(self, do_render: bool = True) -> None:
        if do_render:
            self.render()

        sf_name = "{}/stereo_mixdown.wav".format(self.name)
        orc_name = ".mixdown.orc"
        sco_name = ".mixdown.sco"

        with open(orc_name, "w") as f:
            f.write(self.__make_sampler_orc(n_channels=2))

        sco = " \n".join(
            tuple(
                'i1 0 {} "{}/{}/{}.wav" {} {} {}'.format(
                    self.duration + self.tail,
                    self.name,
                    self.__concatenated_path,
                    track.name,
                    track.volume,
                    track.volume_left,
                    track.volume_right,
                )
                for track in self.orchestration
            )
        )
        with open(sco_name, "w") as f:
            f.write(sco)

        csound.render_csound(sf_name, orc_name, sco_name)

        os.remove(orc_name)
        os.remove(sco_name)
