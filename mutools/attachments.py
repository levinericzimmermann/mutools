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
            head.is_parenthesized = True


class ChooseOne(Attachment):
    name = "choose"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        for head in leaf.note_heads:
            abjad.tweak(head).color = "red"
            head.is_parenthesized = True


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


class ArtificalHarmonic(Attachment):
    name = "artifical_harmonic"
    attach_on_each_part = True
    is_on_off_notation = False

    def __init__(self, nth_harmonic: int) -> None:
        assert nth_harmonic > 1
        self.ground_interval = ji.r(nth_harmonic, 1).cents / 100
        self.harmonic_interval = ji.r(nth_harmonic, nth_harmonic - 1).cents / 100

    def find_pitches(self, abjad_pitch: abjad.NamedPitch) -> None:
        import lily

        basic_pitch = abjad.NamedPitch(
            lily.round_cents_to_12th_tone(float(abjad_pitch) - self.ground_interval)
        )
        harmonic_pitch = abjad.NamedPitch(
            lily.round_cents_to_12th_tone(float(basic_pitch) + self.harmonic_interval)
        )
        return abjad.PitchSegment([basic_pitch, harmonic_pitch])

    def attach(self, leaf: abjad.Chord, novent) -> None:
        try:
            assert len(leaf.note_heads) == 1
        except AssertionError:
            msg = "Artifical harmonics can only be attached to chords that contain "
            msg += "exactly one pitch."
            raise ValueError(msg)

        pitches = self.find_pitches(leaf.note_heads[0].written_pitch)
        leaf.written_pitches = abjad.PitchSegment(pitches)
        abjad.tweak(leaf.note_head[1]).style = "harmonic"


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
        textual_indiciation: str = None,
    ) -> None:
        self.abjad = abjad.MetronomeMark(
            reference_duration=reference_duration,
            units_per_minute=units_per_minute,
            textual_indiciation=textual_indiciation,
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


ALL_ATTACHMENTS = tuple(
    cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if cls[1] != Attachment
)
