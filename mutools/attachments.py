import abc
import inspect
import math
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
        abjad.attach(abjad.LilyPondLiteral("\\once \\tiny"), leaf)
        n_heads = len(leaf.note_heads)

        if n_heads == 1:
            size = 2
            positions2attach = (0,)

        else:
            span = (
                leaf.note_heads[-1].written_pitch.number
                - leaf.note_heads[0].written_pitch.number
            )

            if span <= 7:
                size = 5.45
                positions2attach = (int(math.ceil((n_heads / 2) - 1)),)

            else:
                size = 1.85
                positions2attach = tuple(range(n_heads))

        abjad.attach(
            abjad.LilyPondLiteral(
                "\\once \\override ParenthesesItem.font-size = #{}".format(size)
            ),
            leaf,
        )
        abjad.attach(
            abjad.LilyPondLiteral("\\once \\override ParenthesesItem.padding = #0.05"),
            leaf,
        )

        for position in positions2attach:
            leaf.note_heads[position].is_parenthesized = True

    def attach_first_leaf(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(abjad.LilyPondLiteral("\\startParenthesis"), leaf)
        self.attach(leaf, novent)

    def attach_middle_leaf(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(abjad.LilyPondLiteral("\\once \\tiny"), leaf)

    def attach_last_leaf(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(abjad.LilyPondLiteral("\\endParenthesis"), leaf)
        self.attach(leaf, novent)


class OptionalSomePitches(Optional):
    # optional that is only valid for particular pitches within one harmony

    name = "optional_some_pitches"
    attach_on_each_part = True
    is_on_off_notation = False

    _font_size_optional_note = -2.5

    def __init__(self, optional_pitch_indices: tuple):
        self.optional_pitch_indices = optional_pitch_indices

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(
            abjad.LilyPondLiteral(
                "\\once \\override ParenthesesItem.font-size = #0.75"
            ),
            leaf,
        )
        abjad.attach(
            abjad.LilyPondLiteral("\\once \\override ParenthesesItem.padding = #0.1"),
            leaf,
        )

        for note_idx, note_head in enumerate(leaf.note_heads):
            if note_idx in self.optional_pitch_indices:
                note_head.is_parenthesized = True
                abjad.tweak(note_head).font_size = self._font_size_optional_note
            else:
                abjad.tweak(note_head).font_size = 0

    def attach_middle_leaf(self, leaf: abjad.Chord, novent) -> None:
        for note_idx, note_head in enumerate(leaf.note_heads):
            if note_idx in self.optional_pitch_indices:
                abjad.tweak(note_head).font_size = self._font_size_optional_note
            else:
                abjad.tweak(note_head).font_size = 0


class BartokPizzicato(Attachment):
    name = "bartok_pizzicato"
    attach_on_each_part = False
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(
            abjad.LilyPondLiteral("\\snappizzicato", format_slot="absolute_after"), leaf
        )


class Ornamentation(Attachment):
    name = "ornamentation"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, n_times: int):
        self.n_times = n_times

    @abc.abstractproperty
    def _ornamentation_path() -> str:
        raise NotImplementedError

    @property
    def _markup(self) -> abjad.Markup:
        return abjad.Markup(
            [
                abjad.MarkupCommand("vspace", -0.25),
                abjad.MarkupCommand("fontsize", -4),
                abjad.MarkupCommand("rounded-box", ['{}'.format(self.n_times)]),
                abjad.MarkupCommand("hspace", -0.4),
                abjad.MarkupCommand(
                    "path", 0.25, abjad.LilyPondLiteral(self._ornamentation_path)
                ),
            ],
            direction="up",
        )

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self._markup, leaf)


class OrnamentationUp(Ornamentation):
    _ornamentation_path = """
    #'((moveto 0 0)
      (lineto 0.5 0)
      (curveto 0.5 0 1.5 1.75 2.5 0)
      (lineto 3.5 0))"""


class OrnamentationDown(Ornamentation):
    _ornamentation_path = """
    #'((moveto 0 0)
      (lineto 0.5 0)
      (curveto 0.5 0 1.5 -1.75 2.5 0)
      (lineto 3.5 0))"""


class Choose(Attachment):
    name = "choose"
    attach_on_each_part = False
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        # for head in leaf.note_heads:
        #     abjad.tweak(head).color = "blue"
        abjad.attach(abjad.Markup("\\teeny \\circle +", direction="up"), leaf)


class ChooseOne(Attachment):
    name = "choose"
    attach_on_each_part = False
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        # for head in leaf.note_heads:
        #     abjad.tweak(head).color = "red"
        abjad.attach(abjad.Markup("\\teeny \\circle 1", direction="up"), leaf)


class Tremolo(Attachment):
    name = "tremolo"
    attach_on_each_part = True
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(
            abjad.StemTremolo(32 * (2 ** leaf.written_duration.flag_count)), leaf
        )


class Articulation(Attachment):
    name = "articulation"
    attach_on_each_part = True
    is_on_off_notation = False

    def __init__(self, name: str, direction: str = "up") -> None:
        self.abjad = abjad.Articulation(name, direction=direction)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class ArticulationOnce(Attachment):
    name = "articulation_once"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, name: str, direction: str = "up") -> None:
        self.abjad = abjad.Articulation(name, direction=direction)

    def __eq__(self, other) -> bool:
        try:
            return self.abjad == other.abjad
        except AttributeError:
            return False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Prall(Attachment):
    name = "prall"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self) -> None:
        self.abjad = abjad.LilyPondLiteral('^\\prall', format_slot='after')

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class NaturalHarmonic(Attachment):
    name = "natural_harmonic"
    attach_on_each_part = False
    is_on_off_notation = False

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(
            abjad.LilyPondLiteral("\\flageolet", directed="up", format_slot="after"),
            leaf,
        )


class ArtificalHarmonic(Attachment):
    name = "artifical_harmonic"
    attach_on_each_part = True
    is_on_off_notation = False

    def __init__(self, nth_harmonic: int) -> None:
        assert nth_harmonic > 1
        self.ground_interval = ji.r(nth_harmonic, 1).cents / 100
        self.harmonic_interval = ji.r(nth_harmonic, nth_harmonic - 1).cents / 100

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

    @staticmethod
    def _test_can_be_attached(leaf: abjad.Chord) -> None:
        try:
            assert len(leaf.note_heads) == 1
        except AssertionError:
            msg = "Artifical harmonics can only be attached to chords that contain "
            msg += "exactly one pitch."
            raise ValueError(msg)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        self._test_can_be_attached(leaf)
        pitches = self.find_pitches(leaf.note_heads[0].written_pitch)
        leaf.written_pitches = abjad.PitchSegment(pitches)
        abjad.tweak(leaf.note_heads[1]).style = "harmonic"


class ArtificalHarmonicAddedPitch(ArtificalHarmonic):
    def __init__(self, pitches: abjad.PitchSegment) -> None:
        self.pitches = pitches

    def attach(self, leaf: abjad.Chord, novent) -> None:
        self._test_can_be_attached(leaf)
        leaf.written_pitches = self.pitches
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
        markup = abjad.Markup(self.abjad.markup, direction="up").tiny()
        abjad.attach(markup, leaf)


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


class MarkupOnOff(Attachment):
    name = "markup_on_off"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, *args, **kwargs) -> None:
        self.abjad = abjad.Markup(*args, **kwargs)

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


class _GlissandoAttachment(Attachment):
    name = "__glissando_attachment"

    @staticmethod
    def _set_glissando_layout(
        leaf, thickness: float = 2, minimum_length: float = 2.85
    ) -> None:
        abjad.attach(
            abjad.LilyPondLiteral(
                "\\once \\override Glissando.thickness = #'{}".format(thickness)
            ),
            leaf,
        )
        abjad.attach(
            abjad.LilyPondLiteral(
                "\\once \\override Glissando.minimum-length = #{}".format(
                    minimum_length
                )
            ),
            leaf,
        )
        cmd = "\\once \\override "
        cmd += "Glissando.springs-and-rods = #ly:spanner::set-spacing-rods"
        abjad.attach(abjad.LilyPondLiteral(cmd), leaf)


class _GraceNotesAttachment(_GlissandoAttachment):
    name = "__grace_notes_attachment"
    attach_on_each_part = False
    is_on_off_notation = False

    @staticmethod
    def _attach_grace_not_style(leaf) -> None:
        abjad.attach(
            abjad.LilyPondLiteral('\\once \\override Flag.stroke-style = #"grace"'),
            leaf,
        )


class Acciaccatura(_GraceNotesAttachment):
    name = "acciaccatura"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(
        self, mu_pitches: list, abjad_note: abjad.Note, add_glissando: bool = False
    ):
        self.add_glissando = add_glissando
        self.mu_pitches = mu_pitches
        self.abjad = abjad_note
        self.glissando_minimum_length = 4.15
        self.glissando_thickness = 2
        self.added_attachments = []

    def attach(self, leaf: abjad.Chord, novent) -> None:
        note = abjad.mutate(self.abjad).copy()

        for attachment in self.added_attachments:
            abjad.attach(attachment, note)

        if self.add_glissando:
            abjad.attach(abjad.GlissandoIndicator(), note)
            self._set_glissando_layout(
                note,
                thickness=self.glissando_thickness,
                minimum_length=self.glissando_minimum_length,
            )

        self._attach_grace_not_style(note)

        if type(note) == abjad.Note:
            comparision_pitch_number = note.written_pitch.number
        else:
            comparision_pitch_number = note.note_heads[0].written_pitch.number

        if abs(comparision_pitch_number - leaf.note_heads[0].written_pitch.number) > 8:
            abjad.attach(abjad.LilyPondLiteral("\\grace"), note)
        else:
            abjad.attach(abjad.LilyPondLiteral("\\acciaccatura"), note)
        abjad.attach(abjad.LilyPondLiteral(format(note)), leaf)


class BeforeGraceContainer(_GraceNotesAttachment):
    name = "before_grace_container"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, abjad_notes: tuple, add_glissando: bool = False) -> None:
        self.abjad = abjad.BeforeGraceContainer(abjad_notes)

        if add_glissando:
            for item in self.abjad:
                abjad.attach(abjad.GlissandoIndicator(), item)
            self._set_glissando_layout(self.abjad, thickness=2, minimum_length=2.85)

        self._attach_grace_not_style(self.abjad)
        self.add_glissando = add_glissando

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class BarLine(Attachment):
    name = "bar_line"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, *args, **kwargs) -> None:
        self.abjad = abjad.Fermata(*args, **kwargs)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        abjad.attach(self.abjad, leaf)


class Arpeggio(Attachment):
    name = "arpeggio"
    attach_on_each_part = False
    is_on_off_notation = False

    def __init__(self, direction: int = None) -> None:
        self.abjad = abjad.Arpeggio(direction=direction)

    def attach(self, leaf: abjad.Chord, novent) -> None:
        # hack for avoiding too short arpeggios,
        # see https://code.google.com/archive/p/lilypond/issues/794
        abjad.attach(
            abjad.LilyPondLiteral(
                "\\once \\override Arpeggio #'positions = #(lambda (grob)"
                + " (interval-widen (ly:arpeggio::calc-positions grob) 0.5))"
            ),
            leaf,
        )
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


class Ottava(Attachment):
    name = "ottava"
    attach_on_each_part = False
    is_on_off_notation = True

    def __init__(self, n: int) -> None:
        self.abjad = abjad.Ottava(n, format_slot="before")

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
