from mu.mel import mel
from mu.mel import ji
from mu.utils import tools

from mu.sco import old
from mu.rhy import rhy

try:
    import quicktions as fractions
except ImportError:
    import fractions


"""
def quantisize_rhythm(
    pitch: tuple,
    rhythm: rhy.Compound,
    tick_size: fractions.Fraction = fractions.Fraction(1, 8),
    minima_size=fractions.Fraction(1, 9),
) -> tuple:

    to_small_rhythms = tuple(idx for idx, r in enumerate(rhythm) if r < minima_size)
    pitch = tuple(p for idx, p in enumerate(pitch) if idx not in to_small_rhythms)
    rhythm = tuple(r for idx, r in enumerate(rhythm) if idx not in to_small_rhythms)

    tick_size = fractions.Fraction(tick_size)
    rhythm = tuple(float(r) for r in rhythm)
    duration = fractions.Fraction(sum(rhythm))
    ticks = tuple(tools.frange(0, duration + tick_size, tick_size))
    absolute_rhythm = tools.accumulate_from_zero(rhythm)
    quantisized = [tools.find_closest_item(r, ticks) for r in absolute_rhythm]

    if quantisized[0] != 0:
        quantisized[0] = 0

    try:
        assert len(set(quantisized)) == len(quantisized)

    except AssertionError:
        msg = "Too low 'tick_size' argument for this rhythm. "
        msg += "Stretch rhythm or increase 'tick_size'."
        raise ValueError(msg)

    return pitch, tuple(b - a for a, b in zip(quantisized, quantisized[1:]))
"""


def quantisize_rhythm(
    melody: old.Melody,
    n_divisions: int = 8,
    min_tone_size: fractions.Fraction = 0,
    min_rest_size: fractions.Fraction = fractions.Fraction(1, 10),
) -> tuple:

    new_melody = []

    min_size = fractions.Fraction(1, n_divisions)
    left_over = 0

    for tone in melody:
        r = tone.delay

        if tone.pitch.is_empty:
            is_addable = r >= min_rest_size
        else:
            is_addable = r >= min_tone_size

        if is_addable:
            r += left_over
            left_over = 0
            quantisized = rhy.Unit(round(r * n_divisions) / n_divisions).fraction
            if quantisized == 0:
                quantisized = min_size

            new_tone = tone.copy()
            new_tone.delay = quantisized
            new_tone.duration = quantisized
            new_melody.append(new_tone)

        else:
            left_over += r

    new_melody[-1].delay += left_over
    new_melody[-1].duration += left_over

    return old.Melody(new_melody)


def quantisize_pitches(
    pitches: tuple, scale: tuple, concert_pitch: float = None
) -> ji.JIMel:

    scale = sorted(scale)

    for p in scale:
        assert p.octave == 0

    scale_cent = tuple(p.cents for p in scale)

    quantisized = []
    for pitch in pitches:
        if concert_pitch:
            pitch.concert_pitch_freq = concert_pitch

        if pitch.is_empty:
            quantisized_pitch = mel.TheEmptyPitch
        else:
            octave = pitch.octave
            normalized_cents = pitch.cents + (-octave * 1200)
            quantisized_pitch = scale[
                tools.find_closest_index(normalized_cents, scale_cent)
            ].register(octave)

        quantisized.append(quantisized_pitch)

    return ji.JIMel(quantisized)
