import abc
import itertools

from mu.mel import ji
from mu.mel import mel


class PitchMapTone(object):
    def __init__(self, pitch: ji.JIPitch, weight: float):
        self.__pitch = pitch
        self.__weight = weight

    @property
    def pitch(self):
        return self.__pitch

    @property
    def weight(self):
        return self.__weight

    def set_weight(self, weight):
        copied = self.copy()
        copied.__weight = weight
        return copied

    def __hash__(self):
        return hash((hash(self.pitch), self.weight))

    def __repr__(self):
        return repr((self.pitch, self.weight))

    def copy(self):
        return type(self)(self.pitch, self.weight)


class PitchMapTimeTone(PitchMapTone):
    def __init__(self, pitch: ji.JIPitch, weight: float, time_position: int):
        PitchMapTone.__init__(self, pitch, weight)
        self.time_position = time_position

    def copy(self):
        return type(self)(self.pitch, self.weight, self.time_position)


class AbstractPitchMap(abc.ABC):
    def __init__(self, size=int, harmonicity_function=None):
        if harmonicity_function is None:

            def barlow(p0, p1):
                def c(p0, p1):
                    diff = p1 - p0
                    har = abs(diff.harmonicity_barlow)
                    if har == float("inf"):
                        har = 1
                    return har

                return (c(p0, p1) + c(p1, p0)) / 2

            harmonicity_function = barlow
        self.harmonicity_function = harmonicity_function
        self.size = size

    """
    @abc.abstractproperty
    def _data(self) -> tuple:
        raise NotImplementedError
    """

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_voice(self, voice: list, offset: int = 0):
        raise NotImplementedError

    @abc.abstractmethod
    def devour(self, pitch_map, offset: int = 0) -> "PitchMap":
        raise NotImplementedError

    def calculate_harmonicity_of(self, idx: int):
        item = self[idx]
        item = filter(lambda p: p.pitch != mel.EmptyPitch(), item)
        combinations = itertools.combinations(item, 2)
        harmonicity_weight_pairs = tuple(
            (self.harmonicity_function(p0.pitch, p1.pitch), p0.weight + p1.weight)
            for p0, p1 in combinations
        )
        if harmonicity_weight_pairs:
            return sum(p[0] * p[1] for p in harmonicity_weight_pairs) / sum(
                p[1] for p in harmonicity_weight_pairs
            )
        else:
            return 0

    @property
    def harmonicity(self) -> tuple:
        return tuple(self.calculate_harmonicity_of(idx) for idx in range(len(self)))

    def match(self, idx: int, pitch: ji.JIPitch):
        data = self[idx]
        harmonicity_weight_pairs = []
        for sub in data:
            if sub.pitch != mel.EmptyPitch():
                harmonicity = self.harmonicity_function(sub.pitch, pitch)
                harmonicity_weight_pairs.append((harmonicity, sub.weight))
        full_harmonicity = sum(p[0] * p[1] for p in harmonicity_weight_pairs) / sum(
            p[1] for p in harmonicity_weight_pairs
        )
        return full_harmonicity


class PitchMap(AbstractPitchMap):
    def __init__(self, size=int, harmonicity_function=None):
        AbstractPitchMap.__init__(self, size, harmonicity_function)
        self.__data = list(set([]) for i in range(size))

    @property
    def _data(self):
        return self.__data

    @classmethod
    def from_iterable(cls, iterable, harmonicity_function=None):
        try:
            assert type(iterable) == list
        except AssertionError:
            raise TypeError("Iterable has to be a list")
        for sub in iterable:
            try:
                assert type(sub) == set
            except AssertionError:
                raise TypeError("Iterable has to be filled with sets")

        pm = cls(len(iterable), harmonicity_function)
        pm.__data = iterable
        return pm

    def copy(self):
        def copy_part(part):
            return set([t.copy() for t in part])

        return type(self).from_iterable(
            list(copy_part(p) for p in self), self.harmonicity_function
        )

    def add_tone(self, idx: int, tone: PitchMapTone):
        self.__data[idx].add(tone)

    def add_voice(self, voice: list, offset: int = 0):
        try:
            assert len(voice) + offset <= len(self)
        except AssertionError:
            raise ValueError("Voice doesn't fit in PitchMap. PitchMap is too short!")
        for i, tone in enumerate(voice):
            if tone.pitch != mel.EmptyPitch() and tone.weight > 0:
                self.add_tone(i + offset, tone)

    def devour(self, pitch_map, offset: int = 0) -> "PitchMap":
        res = self.copy()
        try:
            assert len(pitch_map) + offset <= len(self)
        except AssertionError:
            msg = "PitchMap isn't big enough to devour the other PitchMap!"
            raise ValueError(msg)
        for idx, data in enumerate(pitch_map):
            idx += offset
            for tone in data:
                res.add_tone(idx, tone.copy())
        return res


class VoiceDividedPitchMap(AbstractPitchMap):
    def __init__(self, size=int, harmonicity_function=None):
        AbstractPitchMap.__init__(self, size, harmonicity_function)
        self.__voices = set([])

    @property
    def voices(self):
        return self.__voices

    def copy(self):
        def copy_voice(voice):
            return [pmtone.copy() for pmtone in voice]
        voices = [copy_voice(v) for v in self.voices]
        pm = type(self)(self.size, self.harmonicity_function)
        pm.__voices = set(tuple((tuple(v) for v in voices)))
        return pm

    def add_voice(self, voice: list, offset: int = 0):
        try:
            assert len(voice) + offset <= self.size
        except AssertionError:
            raise ValueError("Voice doesn't fit in PitchMap. PitchMap is too short!")
        voice = [PitchMapTone(mel.EmptyPitch(), 0) for i in range(offset)] + list(voice)
        while len(voice) < self.size:
            voice.append(PitchMapTone(mel.EmptyPitch(), 0))
        self.__voices.add(tuple(voice))

    def devour(self, pitch_map, offset: int = 0) -> "VoiceDividedPitchMap":
        res = self.copy()
        try:
            assert len(pitch_map) + offset <= len(self)
        except AssertionError:
            msg = "PitchMap isn't big enough to devour the other PitchMap!"
            raise ValueError(msg)
        for voc in pitch_map.voices:
            res.add_voice(voc, offset)
        return res

    @property
    def _data(self):
        data = [set([]) for i in range(self.size)]
        for v in self.voices:
            for i, sub in enumerate(v):
                if sub.pitch != mel.EmptyPitch():
                    data[i].add(sub)
        return data
