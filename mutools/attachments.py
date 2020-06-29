import abc
import inspect
import sys

import abjad

from mu.mel import ji


class Attachment(abc.ABC):
    @abc.abstractproperty
    def name(cls) -> str:
        raise NotImplementedError

    @abc.abstractproperty
    def attach_on_each_part(cls) -> bool:
        raise NotImplementedError

    @abc.abstractproperty
    def is_on_off_notation(cls) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def attach(self, leaf: abjad.Chord, novent) -> None:
        raise NotImplementedError


class Optional(Attachment):
    name = "optional"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        for head in leaf.note_heads:
            abjad.tweak(head).font_size = -2
            head.is_parenthesized = True


class Choose(Attachment):
    name = "choose"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        for head in leaf.note_heads:
            abjad.tweak(head).color = "blue"


class ChooseOne(Attachment):
    name = "choose"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        for head in leaf.note_heads:
            abjad.tweak(head).color = "red"


class Tremolo(Attachment):
    name = "tremolo"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(abjad.StemTremolo(32), leaf)


class Articulation(Attachment):
    name = "articulation"
    attach_on_each_part = True
    is_on_off_notation = False

    def __init__(self, name: str, direction: str = None) -> None:
        self.abjad = abjad.Articulation(name, direction=direction)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class ArticulationOnce(Attachment):
    name = "articulation_once"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, name: str, direction: str = None) -> None:
        self.abjad = abjad.Articulation(name, direction=direction)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class ArtificalHarmonic(Attachment):
    name = "artifical_harmonic"
    attach_on_each_part = True
    is_on_off_notation = False

    def __init__(self, nth_harmonic: int, notation: abjad.PitchSegment = None) -> None:
        assert nth_harmonic > 1
        self.ground_interval = ji.r(nth_harmonic, 1).cents / 100
        self.harmonic_interval = ji.r(nth_harmonic, nth_harmonic - 1).cents / 100
        self.notation = notation

    def find_pitches(self, abjad_pitch: abjad.NamedPitch) -> None:
        from mutools import lily

        basic_pitch = abjad.NamedPitch(
            lily.round_scale_index_to_12th_tone(
                float(abjad_pitch) - self.ground_interval
            )
        )
        harmonic_pitch = abjad.NamedPitch(
            lily.round_scale_index_to_12th_tone(
                float(basic_pitch) + self.harmonic_interval
            )
        )
        return abjad.PitchSegment([basic_pitch, harmonic_pitch])

    def attach(self, leaf: abjad.Chord, novent) -> None:
        try:
            assert len(leaf.note_heads) == 1
        except AssertionError:
            msg = "Artifical harmonics can only be attached to chords that contain "
            msg += "exactly one pitch."
            raise ValueError(msg)

        if self.notation:
            pitches = self.notation
        else:
            pitches = self.find_pitches(leaf.note_heads[0].written_pitch)
        leaf.written_pitches = abjad.PitchSegment(pitches)
        abjad.tweak(leaf.note_heads[1]).style = "harmonic"


class Dynamic(Attachment):
    name = "dynamic"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, dynamic: str) -> None:
        self.abjad = abjad.Dynamic(dynamic)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class StringContactPoint(Attachment):
    name = "string_contact_point"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, contact_point: str) -> None:
        self.abjad = abjad.StringContactPoint(contact_point)

    def __repr__(self) -> str:
        return repr(self.abjad.contact_point)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad.markup, leaf)


class Tempo(Attachment):
    name = "tempo"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(
        self,
        reference_duration: tuple,
        units_per_minute: tuple,
        textual_indication: str = None,
    ) -> None:
        self.abjad = abjad.MetronomeMark(
            reference_duration=reference_duration,
            units_per_minute=units_per_minute,
            textual_indication=textual_indication,
        )

    def __repr__(self) -> str:
        return repr(self.abjad)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Markup(Attachment):
    name = "markup"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, *args, **kwargs) -> None:
        self.abjad = abjad.Markup(*args, **kwargs)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Fermata(Attachment):
    name = "fermata"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, command: str = "fermata") -> None:
        self.abjad = abjad.Fermata(command)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Arpeggio(Attachment):
    name = "arpeggio"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, direction: int = None) -> None:
        self.abjad = abjad.Arpeggio(direction=direction)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Clef(Attachment):
    name = "clef"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, name: str) -> None:
        self.abjad = abjad.Clef(name)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class MarginMarkup(Attachment):
    name = "margin_markup"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, content: str, context: str = "Staff") -> None:
        cmd = "\\set {}.instrumentName = \\markup ".format(context)
        cmd += "{ " + content + " }"
        self.abjad = abjad.LilyPondLiteral(cmd)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Hauptstimme(Attachment):
    name = "hauptstimme"
    attach_on_each_part = False
    is_on_off_notation = True

    # copied from http://lsr.di.unimi.it/LSR/Snippet?id=843
    _start = abjad.Markup(
        r"""\path #0.25 #'((moveto 0 0)
                 (lineto 0 -2)
                 (moveto 0 -1)
                 (lineto 1 -1)
                 (moveto 1 0)
                 (lineto 1 -2)
                 (moveto 1 0)
                 (lineto 1.8 0))""",
        direction="up",
    )

    _stop = abjad.Markup(
        r"""\path #0.25 #'((moveto 0.7 2.3)
                           (lineto 1.5 2.3)
                           (lineto 1.5 1.4))""",
        direction="up",
    )

    _once = abjad.Markup(
        r"""\path #0.25 #'((moveto 0 0)
                 (lineto 0 -2)
                 (moveto 0 -1)
                 (lineto 1 -1)
                 (moveto 1 0)
                 (lineto 1 -2)
                 (moveto 1 0)
                 (lineto 1.8 0)
                 (moveto 2.5 0)
                 (lineto 3.2 0)
                 (lineto 3.2 -1))""",
        direction="up",
    )

    def __init__(self, is_hauptstimme: bool = False, is_once: bool = False) -> None:
        self.is_hauptstimme = is_hauptstimme
        self.is_once = is_once

    def __eq__(self, other) -> bool:
        try:
            return all(
                (
                    self.is_hauptstimme == other.is_hauptstimme,
                    self.is_once == other.is_once,
                )
            )
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        if self.is_hauptstimme:
            markup = self._start
        else:
            markup = self._stop

        if self.is_once:
            markup = self._once

        abjad.attach(markup, leaf)


ALL_ATTACHMENTS = tuple(
    cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if cls[1] != Attachment
)
