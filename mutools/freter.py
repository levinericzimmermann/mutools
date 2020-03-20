import bisect
import crosstrainer
import functools
import operator
import pylatex

from mu.mel import edo
from mu.mel import ji
from mu.mel import mel


class MonochordFret(object):
    def __init__(self, number, octave) -> None:
        self.number = number
        self.octave = octave

    def __hash__(self):
        return hash((self.number, self.octave))

    def __eq__(self, other):
        try:
            return all((self.number == other.number, self.octave == other.octave))
        except AttributeError:
            return False

    def copy(self):
        return type(self)(int(self.number), int(self.octave))

    def __repr__(self) -> str:
        return "Fret: {0}_{1}".format(self.number, self.octave)

    def convert2absolute_fret(self, divisions) -> int:
        offset = self.octave * divisions
        return self.number + offset


class MonochordString(object):
    def __init__(self, number: int, pitch: ji.JIPitch, divisions=120):
        self.__number = number
        self.pitch = pitch
        self.divisions = divisions
        self.__division_class = edo.EdoPitch.mk_new_edo_class(2, divisions)
        """
        division_floats = [self.__division_class(i) for i in range(divisions)]
        for obj in division_floats:
            obj._concert_pitch = 1
        self.division_floats = [obj.freq for obj in division_floats]
        """
        self.division_floats = MonochordString.mk_division_floats()
        self.fret = self.find_best_fret_for_pitch(pitch)

    def copy(self):
        return type(self)(int(self.number), self.pitch.copy(), self.divisions)

    @staticmethod
    def mk_cents() -> tuple:
        cents = [
            0,
            6,
            15.3,
            23.8,
            32.5,
            42.9,
            50,
            60.7,
            70.5,
            78.5,
            87.5,
            98,
            107,
            116.25,
            126,
            136.5,
            147,
            154,
            164,
            171.5,
            182,
            195,
            205,
            212,
            221.5,
            231,
            241.8,
            252,
            261.5,
            270,
            281,
            290,
            300,
            310,
            319.5,
        ]
        while len(cents) < 120 * 3:
            cents.append(cents[-1] + 10)
        return tuple(cents)

    @staticmethod
    def mk_division_floats() -> tuple:
        cents = MonochordString.mk_cents()
        return tuple(mel.SimplePitch(1, ct).freq for ct in cents)

    @property
    def number(self):
        return self.__number

    def __repr__(self):
        return repr((self.number, self.fret, self.pitch))

    def __eq__(self, other):
        try:
            return all(
                (
                    self.number == other.number,
                    self.pitch == other.pitch,
                    self.fret == other.fret,
                )
            )
        except AttributeError:
            return False

    def find_best_fret_for_pitch(self, pitch):
        octave = 0
        comp = ji.r(2, 1)
        while pitch.float >= comp.float:
            pitch -= comp
            octave += 1
        factor = pitch.float
        closest = bisect.bisect_right(self.division_floats, factor)
        possible_solutions = []
        for c in [closest - 1, closest, closest + 1]:
            try:
                possible_solutions.append(self.division_floats[c])
            except IndexError:
                pass
        hof = crosstrainer.MultiDimensionalRating(size=1, fitness=[-1])
        for pos in possible_solutions:
            if pos > factor:
                diff = pos / factor
            else:
                diff = factor / pos
            hof.append(pos, diff)
        return MonochordFret(self.division_floats.index(hof._items[0]), octave)


def make_string_table(strings: tuple, available_columns=37) -> pylatex.LongTabu:
    import bisect
    from mu.utils import tools

    available_frets = 120 * 3
    frets_per_column = tools.accumulate_from_zero(
        tools.euclid(available_frets, available_columns)
    )
    table = pylatex.LongTabu(
        ("|l| " + "l " * available_columns)[:-1] + "|", row_height=2
    )
    table.add_hline()
    for st_idx, st in enumerate(sorted(strings, key=lambda s: s.number, reverse=True)):
        string_number = len(strings) - st_idx
        fret = st.fret.convert2absolute_fret(120)
        row = ["" for i in range(available_columns)]
        best_column = bisect.bisect_left(frets_per_column, fret)
        if best_column == available_columns:
            best_column -= 1

        if fret % 10 != 0 and fret % 2 == 0:
            color = "red"
        elif fret % 2 != 0 and fret % 5 != 0:
            color = "blue"
        else:
            color = "black"

        col = pylatex.TextColor(color, st.fret.convert2relative_fret())

        row[best_column] = col
        table.add_row(["{0}".format(string_number)] + row)
        table.add_hline()
    return table


def make_table_file(
    name: str,
    strings_per_bar: tuple,
    title_per_bar=None,
    available_columns_per_bar=None,
) -> None:
    if not available_columns_per_bar:
        available_columns_per_bar = tuple(37 for i in strings_per_bar)

    if title_per_bar is None:
        title_per_bar = tuple(
            [
                "Fret positions for bar {0}".format(idx + 1)
                for idx in range(len(strings_per_bar))
            ]
        )
    tables = tuple(
        make_string_table(strings, available_columns=available_columns)
        for strings, available_columns in zip(
            strings_per_bar, available_columns_per_bar
        )
    )
    doc = pylatex.Document(document_options=["landscape", "a4paper"])
    doc.preamble.append(pylatex.Command("usepackage", arguments="lscape"))
    doc.preamble.append(pylatex.Command("usepackage", arguments="xcolor"))
    doc.preamble.append(pylatex.NoEscape(r"\usepackage[a4paper,bindingoffset=0.2in,%"))
    doc.preamble.append(
        pylatex.NoEscape(r"left=0.5cm,right=1cm,top=1.5cm,bottom=1cm,%")
    )
    doc.preamble.append(pylatex.NoEscape(r"footskip=.25in]{geometry}"))
    doc.preamble.append(pylatex.NoEscape(r"\pagenumbering{gobble}"))
    for title, table in zip(title_per_bar, tables):
        # doc.append(pylatex.utils.bold(pylatex.LargeText(title)))
        doc.append(pylatex.LargeText(pylatex.utils.bold(title)))
        doc.append(table)
    doc.generate_pdf(name)
    doc.generate_tex(name)


def find_strings(pitches: tuple) -> tuple:
    return tuple(MonochordString(idx, p) for idx, p in enumerate(pitches))


def find_frets(pitches: tuple) -> tuple:
    return tuple(s.fret for s in find_strings(pitches))


if __name__ == "__main__":
    is_positive = True
    inverse = False

    def mk_pitches(main, side, inverse=False):
        if main == 9:
            primes = (5, 7)
        else:
            primes = (3, 5, 7)
        special = functools.reduce(operator.mul, tuple(p for p in primes if p != main))
        if is_positive and not inverse:
            p0 = ji.r(main, 1)
            p1 = ji.r(main, special)
        elif not is_positive and not inverse:
            p0 = ji.r(main, special)
            p1 = ji.r(main, 1)
        elif is_positive and inverse:
            p0 = ji.r(special, main)
            p1 = ji.r(1, main)
        elif not is_positive and inverse:
            p0 = ji.r(1, main)
            p1 = ji.r(special, main)
        else:
            raise ValueError()

        p0 = p0.normalize(2)
        p1 = p1.normalize(2) + ji.r(4, 1)

        p_between = (ji.r(main, s) for s in side)
        if inverse:
            p_between = (p.inverse() for p in p_between)
        p_inbetween = tuple(p.normalize(2) + ji.r(2, 1) for p in p_between)
        return (p0,) + p_inbetween + (p1,)

    primes = (3, 5, 7, 9)
    pitches = tuple(
        sorted(mk_pitches(p, tuple(s for s in primes if s != p), inverse))
        for p in primes
    )
    pitches = sorted(functools.reduce(operator.add, pitches))
    # print(find_frets(pitches))
    strings = find_strings(pitches)
    make_table_file("FRET_POS_TEST", (strings, strings))
