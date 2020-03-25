"""This modules contains functions that help generating data for pianoteq arguments.
"""

import functools
import operator

from mu.sco import old

from mu.midiplug import midiplug

from mu.mel import ji


class AttributeMaker(object):
    """Class that generates different attributes for pianoteq render."""

    def __init__(
        self,
        voices: tuple,
        metricity_per_beat: tuple = None,
        max_spectrum_profile_change: "dB" = 10,
        dynamic_range: tuple = (0.1, 0.7),
    ) -> None:

        if metricity_per_beat is None:
            metricity_per_beat = tuple(1 for i in range(int(sum(voices[0].delay))))

        iadpptpv = self.detect_intervals_and_duration_pairs_per_tone_per_voice(voices)
        self.__intervals_and_duration_pairs_per_tone_per_voice = iadpptpv
        self.__spectrum_profile_per_tone = self.detect_spectrum_profile_per_tone(
            self.__intervals_and_duration_pairs_per_tone_per_voice,
            max_spectrum_profile_change,
        )

        hatpptpv = self.detect_harmonicity_per_tone_per_voice(
            self.__intervals_and_duration_pairs_per_tone_per_voice
        )
        self.__harmonicity_per_tone_per_voice = hatpptpv
        mptpv = self.detect_metricity_per_tone_per_voice(voices, metricity_per_beat)
        self.__metricity_per_tone_per_voice = mptpv
        vptpv = self.calculate_dynamics_depending_on_metricity_and_harmonicity(
            self.__metricity_per_tone_per_voice,
            self.__harmonicity_per_tone_per_voice,
            dynamic_range,
        )
        self.__volume_per_tone_per_voice = vptpv

    @property
    def spectrum_profile_per_tone(self) -> tuple:
        return self.__spectrum_profile_per_tone

    @property
    def volume_per_tone_per_voice(self) -> tuple:
        return self.__volume_per_tone_per_voice

    @staticmethod
    def detect_metricity_per_tone_per_voice(
        voices: tuple, metricity_per_beat: tuple
    ) -> tuple:
        metricity_per_tone_per_voice = []

        for vox in voices:
            metricity_per_tone_per_voice.append(
                tuple(
                    metricity_per_beat[int(position)]
                    for position in vox.delay.convert2absolute()
                )
            )

        return tuple(metricity_per_tone_per_voice)

    @staticmethod
    def detect_intervals_and_duration_pairs_per_tone_per_voice(voices: tuple) -> tuple:
        voices = tuple(old.Melody(v) for v in voices)
        interval_and_time_pairs_per_tone_per_voice = []
        for v_idx, voice in enumerate(voices):
            poly = old.Polyphon(v for v_idx1, v in enumerate(voices) if v_idx1 != v_idx)
            melody = voice.copy()
            interval_and_time_pairs_per_tone = []
            for tone in melody.convert2absolute():
                start, stop = tone.delay, tone.duration
                simultan_events = functools.reduce(
                    operator.add, tuple(m[:] for m in poly.cut_up_by_time(start, stop))
                )
                interval_and_time_pairs = []
                for simultan_event in simultan_events:
                    if not simultan_event.pitch.is_empty:
                        interval = simultan_event.pitch - tone.pitch
                        interval_and_time_pairs.append(
                            (interval, simultan_event.duration)
                        )
                interval_and_time_pairs_per_tone.append(tuple(interval_and_time_pairs))
            interval_and_time_pairs_per_tone_per_voice.append(
                tuple(interval_and_time_pairs_per_tone)
            )
        return tuple(interval_and_time_pairs_per_tone_per_voice)

    @staticmethod
    def detect_spectrum_profile_per_tone(
        intervals_and_duration_pairs: tuple, max_spectrum_profile_change: float
    ) -> tuple:
        spectrum_profile_per_voice = []
        ig1 = operator.itemgetter(1)

        for voice in intervals_and_duration_pairs:
            prime_counter_per_tone = []
            for tone in voice:
                primes_and_values = [0, 0, 0]
                complete_duration = float(sum(ig1(itp) for itp in tone))
                for interval_time_pair in tone:
                    vector = interval_time_pair[0].monzo
                    while len(vector) < 4:
                        vector += (0,)
                    percentage = float(interval_time_pair[1] / complete_duration)
                    for pidx, prime_exponent in enumerate(vector[1:4]):
                        primes_and_values[pidx] += percentage * prime_exponent
                prime_counter_per_tone.append(tuple(primes_and_values))

            minima_and_maxima_per_prime = tuple(
                (min(p), max(p)) for p in zip(*prime_counter_per_tone)
            )
            range_per_prime = tuple(ma - mi for mi, ma in minima_and_maxima_per_prime)

            dp_minima_and_maxima_per_prime = []
            for minima_and_maxima in minima_and_maxima_per_prime:
                if minima_and_maxima[0] < 0:
                    minima = -max_spectrum_profile_change
                else:
                    minima = 0

                if minima_and_maxima[1] > 0:
                    maxima = max_spectrum_profile_change
                else:
                    maxima = 0

                dp_minima_and_maxima_per_prime.append((minima, maxima))

            dp_range_per_prime = tuple(
                dp_minima_and_maxima[1] - dp_minima_and_maxima[0]
                for dp_minima_and_maxima in dp_minima_and_maxima_per_prime
            )
            spectrum_profile_per_tone = tuple(
                tuple(
                    (dp_range * ((p - min_and_max[0]) / total_range)) + dp_min_max[0]
                    if total_range
                    else 0
                    for p, min_and_max, total_range, dp_min_max, dp_range in zip(
                        primes,
                        minima_and_maxima_per_prime,
                        range_per_prime,
                        dp_minima_and_maxima_per_prime,
                        dp_range_per_prime,
                    )
                )
                for primes in prime_counter_per_tone
            )
            spectrum_profile_per_voice.append(tuple(spectrum_profile_per_tone))

        return tuple(spectrum_profile_per_voice)

    @staticmethod
    def detect_harmonicity_and_time_pairs_per_tone_per_voice(
        interval_and_time_pairs_per_tone_per_voice: tuple
    ) -> tuple:
        return tuple(
            tuple(
                tuple((pair[0].harmonicity_simplified_barlow, pair[1]) for pair in tone)
                for tone in voice
            )
            for voice in interval_and_time_pairs_per_tone_per_voice
        )

    def detect_harmonicity_per_tone_per_voice(
        self, interval_and_time_pairs_per_tone_per_voice: tuple
    ) -> tuple:
        hatpptpv = AttributeMaker.detect_harmonicity_and_time_pairs_per_tone_per_voice(
            interval_and_time_pairs_per_tone_per_voice
        )
        harmonicity_per_tone_per_voice = []
        ig1 = operator.itemgetter(1)
        for voice in hatpptpv:
            newv = []
            for tone in voice:
                durations = tuple(ig1(s) for s in tone)
                complete_duration = sum(durations)
                newv.append(sum(h * (d / complete_duration) for h, d in tone))
            harmonicity_per_tone_per_voice.append(tuple(newv))
        return tuple(harmonicity_per_tone_per_voice)

    @staticmethod
    def calculate_dynamics_depending_on_metricity_and_harmonicity(
        metricity_per_tone_per_voice: tuple,
        harmonicity_per_tone_per_voice: tuple,
        dynamic_range: tuple,
    ) -> tuple:
        min_vol = dynamic_range[0]
        absolute_dynamic_range = dynamic_range[1] - dynamic_range[0]
        vol_per_voice = []
        for metric_strength_per_tone, harmonicity_per_tone in zip(
            metricity_per_tone_per_voice, harmonicity_per_tone_per_voice
        ):
            sorted_harmonicity = sorted(set(harmonicity_per_tone))
            n_harmonicities = len(sorted_harmonicity) + 1
            normalized_harmonicity = tuple(
                (sorted_harmonicity.index(h) + 1) / n_harmonicities
                for h in harmonicity_per_tone
            )
            volumes = []
            for harmonicity, metricity in zip(
                normalized_harmonicity, metric_strength_per_tone
            ):
                volume = min_vol + (
                    absolute_dynamic_range * ((harmonicity * 0.45) + (metricity * 0.65))
                )
                volumes.append(volume)
            vol_per_voice.append(tuple(volumes))
        return tuple(vol_per_voice)


def mk_empty_attack(
    duration: float,
    volume: float,
    frequencey: float = 20,
    hammer_noise: float = 3,
    impedance: float = 0.3,
    cutoff: float = 0.3,
    q_factor: float = 5,
    string_length: float = 0.8,
    strike_point: float = 1 / 2,
    hammer_hard_piano: float = None,
    hammer_hard_mezzo: float = 1,
    hammer_hard_forte: float = 2,
    blooming_energy=None,
) -> midiplug.PyteqTone:
    """Helps making percussive sounds with Pianoteq."""
    return midiplug.PyteqTone(
        ji.JIPitch(ji.r(1, 1), multiply=frequencey),
        duration,
        duration,
        volume=volume,
        hammer_noise=hammer_noise,
        impedance=impedance,
        cutoff=cutoff,
        q_factor=q_factor,
        string_length=string_length,
        strike_point=strike_point,
        hammer_hard_piano=hammer_hard_piano,
        hammer_hard_mezzo=hammer_hard_mezzo,
        hammer_hard_forte=hammer_hard_forte,
        blooming_energy=blooming_energy,
    )
