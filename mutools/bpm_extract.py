"""This module collects different methods of detecting the BPM of a file."""

import aubio
import essentia.standard as es
import librosa
import pyo

import numpy as np


class BPM(float):
    _available_methods = set(("essentia", "aubio", "librosa"))

    def __new__(self, path: str, method: str = "librosa", params: dict = None):
        assert method in self._available_methods
        bpm = getattr(self, "_{}".format(method))(path, params)
        return super().__new__(self, bpm)

    @staticmethod
    def _beats_to_bpm(beats: float, path: str = None) -> float:
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in {:s}".format(path))
            bpms = 60.0 / np.diff(beats)
            return np.median(bpms)
        else:
            print("not enough beats found in {:s}".format(path))
            return 0

    @staticmethod
    def _aubio(path: str, params=None) -> float:
        """Using aubio to calculate the bpm of a given files.

        This function is copied from the example files of aubio and
        could be found here:
            https://github.com/aubio/aubio/blob/master/python/demos/demo_bpm_extract.py

        The arguments are:
            path: path to the file
            param: dictionary of parameters
        """
        if params is None:
            params = {}

        # default:
        samplerate, win_s, hop_s = 44100, 1024, 512
        if "mode" in params:
            if params.mode in ["super-fast"]:
                # super fast
                samplerate, win_s, hop_s = 4000, 128, 64
            elif params.mode in ["fast"]:
                # fast
                samplerate, win_s, hop_s = 8000, 512, 128
            elif params.mode in ["default"]:
                pass
            else:
                raise ValueError("unknown mode {:s}".format(params.mode))

        # manual settings
        if "samplerate" in params:
            samplerate = params.samplerate

        if "win_s" in params:
            win_s = params.win_s

        if "hop_s" in params:
            hop_s = params.hop_s

        s = aubio.source(path, samplerate, hop_s)
        samplerate = s.samplerate
        o = aubio.tempo("specdiff", win_s, hop_s, samplerate)
        # List of beats, in samples
        beats = []
        # Total number of frames read
        total_frames = 0

        while True:
            samples, read = s()
            is_beat = o(samples)
            if is_beat:
                this_beat = o.get_last_s()
                beats.append(this_beat)
                # if o.get_confidence() > .2 and len(beats) > 2.:
                #    break
            total_frames += read
            if read < hop_s:
                break

        return BPM._beats_to_bpm(beats, path)

    @staticmethod
    def _essentia(path: str, params=None) -> float:
        """Using essentia to calculate the bpm of a given files.

        This function has been copied from the essentia examples here:
            https://essentia.upf.edu/essentia_python_examples.html
        """

        info = pyo.sndinfo(path)

        audio = es.MonoLoader(filename=path, sampleRate=info[2])()

        # Compute beat positions and BPM
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        return bpm

    @staticmethod
    def _librosa(path: str, params=None) -> float:
        y, sr = librosa.load(path)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return float(tempo[0])
