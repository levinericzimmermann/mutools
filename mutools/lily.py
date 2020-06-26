import abjad

from mu.mel import ji
from mu.sco import old
from mu.utils import tools

from . import attachments

"""This module contains small functions that may help to generate scores with abjad.

The main focus are functions that may help to convert algorithmically generated abstract
musical data to a form thats closer to notation (where suddenly questions about
time signature, beams, ties and accidentals are occuring).
"""


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

    def append(self, novent: NOvent) -> None:
        try:
            assert novent.delay < novent.duration
        except AssertionError:
            msg = "NOvent has to have absolute time representation "
            msg += "to be able to be added."
            raise TypeError(msg)

        try:
            assert not self.is_occupied(novent.delay, novent.duration)
        except AssertionError:
            msg = "Position is already occupied!"
            raise ValueError(msg)

        self._novents.append(novent)

    @property
    def _sorted_novents(self) -> list:
        return sorted(self._novents, key=lambda novent: novent.delay)

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

    def is_occupied(self, start: float, end: float) -> bool:
        for novent in self._novents:
            if not (novent.duration <= start or end <= novent.delay):
                return True

        return False

    def __iter__(self) -> iter:
        return iter(self._sorted_novents)


def mk_no_time_signature():
    return abjad.LilyPondLiteral("override Score.TimeSignature.stencil = ##f", "before")


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


"""
def convert_abjad_pitches_and_mu_rhythms2abjad_notes(
    harmonies: list, delays: list, grid
) -> list:
    leading_pulses = grid.leading_pulses
    absolute_leading_pulses = tuple(itertools.accumulate([0] + list(leading_pulses)))
    converted_delays = grid.apply_delay(delays)
    absolute_delays = tuple(itertools.accumulate([0] + list(converted_delays)))
    # 1. generate notes
    notes = abjad.Measure(abjad.TimeSignature(grid.absolute_meter), [])
    resulting_durations = []
    for harmony, delay, start, end in zip(
        harmonies, converted_delays, absolute_delays, absolute_delays[1:]
    ):
        subnotes = abjad.Voice()
        seperated_by_grid = seperate_by_grid(
            delay, start, end, absolute_leading_pulses, leading_pulses, grid
        )
        assert sum(seperated_by_grid) == delay
        for d in seperated_by_grid:
            seperated_by_assignable = seperate_by_assignability(d, grid)
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
    # 2. apply beams
    return apply_beams(notes, resulting_durations, absolute_leading_pulses)
"""


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
    octave = pitch.octave + 4
    pitch_class = ratio2pitchclass_dict[pitch.register(0)]
    confused_octave_tests = (pitch_class[0] == "c", pitch.register(0).cents > 1000)
    if all(confused_octave_tests):
        octave += 1
    return abjad.NamedPitch(pitch_class, octave=octave)


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
  (-1/12 #xE2C2)
  (-1/6 #xE2CC)
  (-1/3 #xE2D0)
  (-5/12 #xE2C6)
  (-7/12 #xE2C1)
  (-2/3 #xE2CB)
  (-5/6 #xE2CF)
  (-11/12 #xE2C5))

"""
