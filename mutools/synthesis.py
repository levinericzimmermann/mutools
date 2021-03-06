import abc
import os
import sox
import subprocess
import uuid

import pyo64 as pyo

from mu.mel import ji
from mu.mel import mel
from mu.sco import old

from mu.midiplug import midiplug

from mutools import csound


class SoundEngine(abc.ABC):
    CONCERT_PITCH = 260

    @abc.abstractmethod
    def render(self, name: str) -> None:
        """Generate WAV file with input name."""
        raise NotImplementedError


class BasedCsoundEngine(SoundEngine):
    print_output = False
    remove_files = True
    sr = 96000

    @abc.abstractmethod
    def cname(self) -> str:
        raise NotImplementedError

    @abc.abstractproperty
    def orc(self) -> str:
        raise NotImplementedError

    @abc.abstractproperty
    def sco(self) -> str:
        raise NotImplementedError

    def render(self, name: str) -> subprocess.Popen:
        id_number = uuid.uuid4().hex
        orc_name = "{}_{}.orc".format(self.cname, id_number)
        sco_name = "{}_{}.sco".format(self.cname, id_number)

        for fname, data in ((orc_name, self.orc), (sco_name, self.sco)):
            with open(fname, "w") as f:
                f.write(data)

        return csound.render_csound(
            "{}.wav".format(name),
            orc_name,
            sco_name,
            print_output=self.print_output,
            remove_files=self.remove_files,
            sr=self.sr,
        )


class SimpleCsoundSinePlayer(BasedCsoundEngine):
    cname = ".simple_sine"

    default_volume = 0.7
    concert_pitch = 260
    tail = 1

    def __init__(self, melody: old.Melody):
        self._melody = melody

    @property
    def orc(self) -> str:
        lines = [
            "0dbfs=1",
            "gaSendL, gaSendR init 0",
            "nchnls=1\n",
            "instr 1",
            "kvol linseg 0, 0.1, 1, p3 - 0.2, 1, 0.1, 0",
            "asig poscil3 kvol * p5, p4",
            "gaSendL  =        gaSendL + asig/3",
            "gaSendR  =        gaSendR + asig/3",
            "out asig",
            "endin\n",
            "instr 100",
            "aRvbL, aRvbR reverbsc gaSendL,gaSendR,0.6,7000",
            "out     (aRvbL + aRvbR) * 0.15",
            "clear    gaSendL,gaSendR",
            "endin\n",
        ]
        return "\n".join(lines)

    @property
    def sco(self) -> str:
        lines = ["i100 0 {}".format(float(self._melody.duration) + self.tail)]
        for tone, tone_rel in zip(self._melody.convert2absolute(), self._melody):
            if not tone.pitch.is_empty:
                volume = tone.volume
                if not volume:
                    volume = self.default_volume
                lines.append(
                    "i1 {} {} {} {}".format(
                        float(tone.delay),
                        float(tone_rel.duration),
                        float(tone.pitch) * self.concert_pitch,
                        volume,
                    )
                )
        return "\n".join(lines)


class SilenceEngine(BasedCsoundEngine):
    def __init__(self, duration: float) -> None:
        self.__duration = duration

    @property
    def cname(self) -> str:
        return ".silence"

    @property
    def duration(self) -> float:
        return self.__duration

    @property
    def orc(self) -> str:
        lines = (r"0dbfs=1", r"instr 1", r"asig poscil 0, 100", r"out asig", r"endin")
        return "\n".join(lines)

    @property
    def sco(self) -> str:
        lines = (r"i1 0 {}".format(self.duration),)
        return "\n".join(lines)


class PyoEngine(SoundEngine):
    def __init__(self):
        self.__server = pyo.Server(sr=96000, audio="offline", nchnls=1).boot()

    @abc.abstractproperty
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def server(self) -> pyo.Server:
        return self.__server

    @abc.abstractmethod
    def render(self, name: str) -> None:
        raise NotImplementedError

    def copy(self) -> "PyoEngine":
        return type(self)()


class PyoMelodyEngine(PyoEngine):
    def __init__(
        self,
        melody: old.Melody,
        attack: float = 0.025,
        decay: float = 0.05,
        sustain: float = 1,
        release: float = 0.025,
    ):
        super().__init__()
        self.__melody = melody
        self.__duration = float(melody.duration)
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    @property
    def duration(self) -> float:
        return self.__duration

    @abc.abstractproperty
    def instrument(self) -> pyo.EventInstrument:
        raise NotImplementedError

    def copy(self) -> "PyoMelodyEngine":
        return type(self)(self.__melody)

    def render(self, name: str) -> None:
        self.server.recordOptions(
            dur=self.duration, filename="{}.wav".format(name), sampletype=4
        )
        freq = tuple(
            tone.pitch.freq if not tone.pitch.is_empty else 0 for tone in self.__melody
        )
        dur = tuple(float(tone.duration) for tone in self.__melody)

        events = pyo.Events(
            instr=self.instrument,
            freq=freq,
            dur=dur,
            attack=self.attack,
            decay=self.decay,
            sustain=self.sustain,
            release=self.release,
        )
        events.play()
        self.server.start()


class SineMelodyEngine(PyoMelodyEngine):
    @property
    def instrument(self) -> pyo.EventInstrument:
        class SinePlayer(pyo.EventInstrument):
            def __init__(self, **args) -> None:
                pyo.EventInstrument.__init__(self, **args)
                self.osc = pyo.Sine(freq=self.freq, mul=self.env).out(dur=self.dur)

        return SinePlayer


class PyteqEngine(SoundEngine):
    def __init__(
        self, preset=None, fxp=None, available_midi_notes=tuple(range(128)), volume=0.5
    ):
        self.__volume = volume
        self.__available_midi_notes = available_midi_notes
        self.__preset = preset
        self.__fxp = fxp

    @property
    def volume(self) -> float:
        return self.__volume

    @property
    def preset(self) -> str:
        return self.__preset

    @property
    def fxp(self) -> str:
        return self.__fxp

    @property
    def available_midi_notes(self) -> tuple:
        return self.__available_midi_notes

    def render(self, name: str, cadence: old.Cadence) -> subprocess.Popen:
        seq = []
        for chord in cadence:
            dur = float(chord.delay)
            if chord.pitch != mel.TheEmptyPitch and bool(chord.pitch):
                size = len(chord.pitch)
                for idx, pi in enumerate(chord.pitch):
                    if idx + 1 == size:
                        de = float(dur)
                    else:
                        de = 0
                    if pi != mel.TheEmptyPitch:
                        if chord.volume:
                            volume = chord.volume
                        else:
                            volume = self.volume

                        tone = midiplug.PyteqTone(
                            ji.JIPitch(pi, multiply=self.CONCERT_PITCH),
                            de,
                            dur,
                            volume=volume,
                        )
                    else:
                        tone = midiplug.PyteqTone(
                            mel.TheEmptyPitch, de, dur, volume=self.volume
                        )
                    seq.append(tone)
            else:
                seq.append(old.Rest(dur))

        pt = midiplug.Pianoteq(tuple(seq), self.available_midi_notes)
        return pt.export2wav(name, 1, self.preset, self.fxp)


class CsoundEngine(SoundEngine):
    @abc.abstractproperty
    def orc(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def mk_sco(self, cadence: old.Cadence) -> str:
        raise NotImplementedError

    def render(self, name: str, cadence: old.Cadence) -> None:
        sfname = "{0}.wav".format(name)
        fname = "csoundsynth"
        orc_name = "{0}.orc".format(fname)
        sco_name = "{0}.sco".format(fname)
        sco = self.mk_sco(cadence)
        if sco:
            with open(orc_name, "w") as f:
                f.write(self.orc)
            with open(sco_name, "w") as f:
                f.write(sco)
            cmd0 = "csound --format=double -k 96000 -r 96000 -o {0} ".format(sfname)
            cmd1 = "{0} {1}".format(orc_name, sco_name)
            cmd = cmd0 + cmd1
            os.system(cmd)
            os.remove(orc_name)
            os.remove(sco_name)


class SampleEngine(CsoundEngine):
    """pitch2sample has to be a dict with the following structure:

    {pitch0: CYCLE((SAMPLE_NAME, PITCH_FACTOR), (SAMPLE_NAME, PITCH_FACTOR), ...),
     pitch1: CYCLE((SAMPLE_NAME, PITCH_FACTOR), ...),
     ...
     pitchN: CYCLE((SAMPLE_NAME, PITCH_FACTOR), ...)}
    """

    def __init__(self, pitch2sample: dict) -> None:
        self.__pitch2sample = pitch2sample

    @property
    def pitch2sample(self) -> dict:
        return self.__pitch2sample

    @property
    def orc(self) -> str:
        lines = (
            r"0dbfs=1",
            r"gaSend init 0",
            r"instr 1",
            r"asig diskin2 p4, p5, 0, 0, 6, 4",
            r"out asig * p6",
            r"gaSend = gaSend + (asig * 0.1)",
            r"endin",
            r"instr 2",
            r"kroomsize init 0.7",
            r"kHFDamp init 0.5",
            r"aRvbL, aRvbR freeverb gaSend, gaSend, kroomsize, kHFDamp",
            r"out (aRvbL + aRvbR) * 0.4",
            r"clear gaSend",
            r"endin",
        )
        return "\n".join(lines)

    def mk_sco(self, cadence: old.Cadence) -> str:
        lines = []
        abs_start = cadence.delay.convert2absolute()
        for event, start in zip(cadence, abs_start):
            if event.pitch and event.pitch != mel.TheEmptyPitch:
                line = r"i1 {0}".format(start)
                for pi in event.pitch:
                    if pi != mel.TheEmptyPitch:
                        s_info = next(self.pitch2sample[pi])
                        sample_name, factor = s_info[0], s_info[1]
                        if len(s_info) == 3:
                            vol = s_info[2]
                        else:
                            vol = 1
                        duration = sox.file_info.duration(sample_name)
                        final_line = '{0} {1} "{2}" {3} {4}'.format(
                            line, duration, sample_name, factor, vol
                        )
                        lines.append(final_line)
        complete_duration = float(cadence.duration + 5)
        lines.append("i2 0 {0}".format(complete_duration))
        return "\n".join(lines)
