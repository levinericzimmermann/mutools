"""MUS is an extension of MU for organising compositions with instruments & mixed media.

Equal to MU, One MUS.MU object shall represent one piece. It is composed of different
Segment objects.

For the usual workflow it is expected that the user...
    (1) first defines an Orchestration class for the complete piece.
    (2) then defines classes that inherit from the MUS.Segment class. To overwrite
        the default render method is especially crucial for this process.
    (3) last but not least, build objects from those Segment classes
        that are the input for the MU object.

Then single tracks can be rendered through MUs render method.
"""

from mutools import MU


class Track(MU.Track):
    def __init__(self, name: str, n_staves: int = 1, **kwargs):
        super().__init__(name, **kwargs)
        self.n_staves = n_staves


class Segment(MU.Segment):
    def __init__(self, name: str, **kwargs) -> None:
        super.__init__(name, **kwargs)

    def render(self, path: str, sound: bool = True, score: bool = True) -> tuple:
        if sound:
            super().render(path)

        if score:
            pass


class MU(MU.MU):
    def render(self, score: bool = True, sound: bool = True) -> None:
        if sound:
            super().render()

    def render_segment(
        self, segment_name: str, score: bool = True, sound: bool = True
    ) -> None:
        if sound:
            super().render_segment(segment_name)
