from mu.mel import mel
from mu.mel import ji
from mu.rhy import rhy
from mu.utils import tools

try:
    import quicktions as fractions
except ImportError:
    import fractions


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
