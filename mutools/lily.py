"""This module contains small functions that may help to generate scores with abjad.

The main focus are functions that may help to convert algorithmically generated abstract
musical data to a form thats closer to notation (where suddenly questions about
time signature, beams, ties and accidentals are occuring).
"""

import subprocess

import quicktions as fractions

import abjad

from mu.mel import ji
from mu.sco import old
from mu.utils import tools

from . import attachments


class NOvent(old.Ovent):
    _available_attachments = {at.name: at for at in attachments.ALL_ATTACHMENTS}

    def __init__(self, *args, **kwargs) -> None:
        new_kwargs = {}

        obj_attachments = {name: None for name in self._available_attachments}
        for kwarg in kwargs:
            if kwarg in obj_attachments:
                obj_attachments.update({kwarg: kwargs[kwarg]})
            else:
                new_kwargs.update({kwarg: kwargs[kwarg]})

        for attachment in obj_attachments:
            setattr(self, attachment, obj_attachments[attachment])

        super().__init__(*args, **new_kwargs)

    @property
    def attachments(self) -> tuple:
        at = tuple(
            filter(
                lambda x: bool(x) and x.name != "artifical_harmonic",
                (getattr(self, name) for name in self._available_attachments),
            )
        )
        if self.artifical_harmonic:
            at = (self.artifical_harmonic,) + at

        return at

    @classmethod
    def _get_standard_attributes(cls) -> tuple:
        return tuple(
            filter(lambda x: x != "attachments", tools.find_attributes_of_object(cls()))
        )


class NOventLine(old.OventLine):
    """A NOventLine contains sequentially played NOvents."""

    _object = NOvent()


class NOventSet(object):
    def __init__(self, *novent, size: float = None) -> None:
        self._novents = list(novent)
        self._size = size

    def append(self, novent: NOvent, ignore_rests: bool = True) -> None:
        try:
            assert novent.delay < novent.duration
        except AssertionError:
            msg = "NOvent has to have absolute time representation "
            msg += "to be able to be added."
            raise TypeError(msg)

        try:
            assert not self.is_occupied(novent.delay, novent.duration, ignore_rests)
        except AssertionError:
            msg = "Position is already occupied!"
            raise ValueError(msg)

        self._novents.append(novent)

    @property
    def _sorted_novents(self) -> list:
        return sorted(self._novents, key=lambda novent: novent.delay)

    @property
    def size(self) -> fractions.Fraction:
        return self._size

    @property
    def novent_line(self) -> NOventLine:
        sorted_novents = self._sorted_novents
        novent_line = NOventLine([], time_measure="absolute")

        if self._novents:
            if sorted_novents[0].delay > 0:
                sorted_novents.insert(
                    0, NOvent(pitch=[], delay=0, duration=sorted_novents[0].delay)
                )

            for novent0, novent1 in zip(sorted_novents, sorted_novents[1:]):
                novent_line.append(novent0)
                diff = novent1.delay - novent0.duration
                if diff > 0:
                    novent_line.append(
                        NOvent(pitch=[], delay=novent0.duration, duration=novent1.delay)
                    )

            novent_line.append(sorted_novents[-1])

            if self._size:
                diff = self._size - novent_line[-1].duration
                if diff > 0:
                    novent_line.append(
                        NOvent(
                            pitch=[],
                            delay=novent_line[-1].duration,
                            duration=self._size,
                        )
                    )

        else:
            if self._size:
                novent_line.append(NOvent(pitch=[], delay=0, duration=self._size))

        return novent_line.convert2relative()

    def is_occupied(self, start: float, end: float, ignore_rests: bool = False) -> bool:
        for novent in self._novents:
            if not all((ignore_rests, not novent.pitch)):
                if not (novent.duration <= start or end <= novent.delay):
                    return True

        return False

    def __iter__(self) -> iter:
        return iter(self._sorted_novents)

    def detect_undefined_areas(self, ignore_rests: bool = False) -> tuple:
        sorted_novents = self._sorted_novents
        if ignore_rests:
            sorted_novents = tuple(novent for novent in sorted_novents if novent.pitch)

        undefined_areas = []

        if sorted_novents[0].delay != 0:
            undefined_areas.append((0, fractions.Fraction(sorted_novents[0].delay)))

        for novent0, novent1 in zip(sorted_novents, sorted_novents[1:]):
            if novent0.duration != novent1.delay:
                undefined_areas.append(
                    (
                        fractions.Fraction(novent0.duration),
                        fractions.Fraction(novent1.delay),
                    )
                )

        if self.size:
            if sorted_novents[-1].duration < self.size:
                undefined_areas.append(
                    (
                        fractions.Fraction(sorted_novents[-1].duration),
                        fractions.Fraction(self.size),
                    )
                )

        return tuple(undefined_areas)


def mk_no_time_signature():
    return abjad.LilyPondLiteral(
        "\\override Score.TimeSignature.stencil = ##f", "before"
    )


def mk_numeric_ts() -> abjad.LilyPondLiteral:
    return abjad.LilyPondLiteral("numericTimeSignature", "before")


def mk_staff(voices, clef="percussion") -> abjad.Staff:
    staff = abjad.Staff([])
    for v in voices:
        staff.append(v)
    clef = abjad.Clef(clef)
    abjad.attach(clef, staff)
    abjad.attach(mk_numeric_ts(), staff[0][0])
    return staff


def mk_cadenza():
    return abjad.LilyPondLiteral("\\cadenzaOn", "before")


def mk_bar_line() -> abjad.LilyPondLiteral:
    return abjad.LilyPondLiteral('bar "|"', "after")


def round_scale_index_to_12th_tone(index: float) -> float:
    # round to 12th tone
    ct = round(index * 6) / 6
    return ct


def round_cents_to_12th_tone(cents: float) -> float:
    ct = cents / 100
    return round_scale_index_to_12th_tone(ct)


def convert2abjad_pitch(
    pitch: ji.JIPitch, ratio2pitchclass_dict: dict
) -> abjad.NamedPitch:
    """Simple function to convert mu.mel.JIPitch to abjad.NamedPitch"""
    octave = pitch.octave + 4
    pitch_class = ratio2pitchclass_dict[pitch.register(0)]

    confused_octave_tests = (pitch_class[0] == "c", pitch.register(0).cents > 1000)
    if all(confused_octave_tests):
        octave += 1

    return abjad.NamedPitch(pitch_class, octave=octave)


def make_small_example(
    score: abjad.Score,
    path: str,
    size: float = None,
    staff_size: float = 20,
    resolution: int = 500,
    header_block=abjad.Block("header"),
) -> subprocess.Popen:
    includes = ["lilypond-book-preamble.ly"]

    score_block = abjad.Block("score")
    score_block.items.append(score)

    layout_block = abjad.Block("layout")
    layout_block.items.append(r"indent = 0\mm")
    layout_block.items.append(r"short-indent = 0\mm")
    layout_block.items.append(r"ragged-last = ##f")
    layout_block.items.append(r"ragged-right = ##f")

    lilypond_file = abjad.LilyPondFile(
        lilypond_version_token=abjad.LilyPondVersionToken(LILYPOND_VERSION),
        global_staff_size=staff_size,
        includes=includes,
        items=[layout_block, header_block, score_block],
    )
    write_lily_file(lilypond_file, path)
    return render_lily_file(
        path, write2png=True, resolution=resolution, output_name=path
    )


def write_lily_file(lilypond_file: abjad.LilyPondFile, path: str) -> None:
    with open("{}.ly".format(path), "w") as f:
        f.write(EKMELILY_PREAMBLE)
        f.write(START_END_PARENTHESIS)
        f.write(format(lilypond_file))


def render_lily_file(
    lilyfile_path: str,
    write2png: bool = False,
    resolution: int = 500,
    output_name: str = None,
) -> subprocess.Popen:
    cmd = ["lilypond"]
    if write2png:
        cmd.extend(["--png", "-dresolution={}".format(resolution)])
    if output_name:
        cmd.append("-o{}".format(output_name))
    cmd.append("{}.ly".format(lilyfile_path))

    return subprocess.Popen(cmd)


EKMELILY_PREAMBLE = """
\\include "ekmel.ily"
\\language "english"
\\ekmelicStyle "gost"

\\ekmelicUserStyle pfeifer #'(
  (1/12 #xE2C7)
  (1/6 #xE2D1)
  (1/3 #xE2CD)
  (5/12 #xE2C3)
  (7/12 #xE2C8)
  (2/3 #xE2D2)
  (5/6 #xE2CE)
  (11/12 #xE2C4)
  (13/12 #xE2C9)
  (7/6 #xE2D3)
  (-1/12 #xE2C2)
  (-1/6 #xE2CC)
  (-1/3 #xE2D0)
  (-5/12 #xE2C6)
  (-7/12 #xE2C1)
  (-2/3 #xE2CB)
  (-5/6 #xE2CF)
  (-11/12 #xE2C5)
  (-13/12 #xE2C0)
  (-7/6 #xE2CA))
"""


START_END_PARENTHESIS = r"""
startParenthesis = {
  \once \override ParenthesesItem.stencils = #(lambda (grob)
        (let ((par-list (parentheses-item::calc-parenthesis-stencils grob)))
          (list (car par-list) point-stencil )))
}

endParenthesis = {
  \once \override ParenthesesItem.stencils = #(lambda (grob)
        (let ((par-list (parentheses-item::calc-parenthesis-stencils grob)))
          (list point-stencil (cadr par-list))))
}
"""


LILYPOND_VERSION = "2.19.83"
