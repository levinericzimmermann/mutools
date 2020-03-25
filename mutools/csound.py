"""This module contains small functions for using csound within Python"""

import subprocess


def render_csound(
    file_name: str, orc_name: str, sco_name: str, print_output: bool = False
) -> None:
    cmd = ["csound", "--format=double", "-k 96000", "-r 96000"]

    if print_output is False:
        cmd.append("-O null")

    cmd.extend(["-o {0}".format(file_name), orc_name, sco_name])

    subprocess.call(" ".join(cmd), shell=True)
