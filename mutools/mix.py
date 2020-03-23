import os
import shutil
import sox


def mix_mono(outputname, *inputname) -> None:
    outputname = "{0}.wav".format(outputname)
    inputname = tuple("{0}.wav".format(n) for n in inputname)

    size = len(inputname)
    if size > 1:
        cbn = sox.Combiner()
        cbn.build(list(inputname), outputname, "merge")
    elif size == 1:
        shutil.copyfile(inputname[0], outputname)
    else:
        with open(outputname, "w") as f:
            f.write("")


def mix_complex(outputname, *inputdata):
    """one inputdata consist of three arguments:

        (FILENAME, VOLUME, PAN)
    """

    outputname = "{0}.wav".format(outputname)

    def mk_orc():
        lines = (
            r"0dbfs=1",
            r"nchnls=2",
            r"instr 1",
            r"asig diskin2 p4, 1, 0, 0, 6, 4",
            r"asig = asig * p5",
            r"outs asig * p6, asig * p7",
            r"endin",
        )
        return " \n".join(lines)

    def mk_sco(inputdata):
        def get_panning_arguments(pan) -> tuple:
            return 1 - pan, pan

        lines = []
        for f in inputdata:
            name, volume, pan, start = f
            name = "{0}.wav".format(name)
            pan0, pan1 = get_panning_arguments(pan)
            duration = sox.file_info.duration(name)
            line = 'i1 {5} {0} "{1}" {2} {3} {4}'.format(
                duration, name, volume, pan0, pan1, start
            )
            lines.append(line)
        return " \n".join(lines)

    fname = "complexMix"
    orc_name = "{0}.orc".format(fname)
    sco_name = "{0}.sco".format(fname)
    orc = mk_orc()
    sco = mk_sco(inputdata)
    for n, content in ((orc_name, orc), (sco_name, sco)):
        with open(n, "w") as f:
            f.write(content)

    cmd0 = "csound --format=double -k 96000 -r 96000 -o {0} ".format(outputname)
    cmd1 = "{0} {1}".format(orc_name, sco_name)
    cmd = cmd0 + cmd1
    os.system(cmd)
    os.remove(orc_name)
    os.remove(sco_name)
