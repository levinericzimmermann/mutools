"""This module contains small functions for using csound within Python"""

import subprocess


def render_csound(
    file_name: str,
    orc_name: str,
    sco_name: str,
    print_output: bool = False,
    remove_files: bool = True,
) -> subprocess.Popen:
    cmd = ["csound", "--format=double", "-k 96000", "-r 96000"]

    if print_output is False:
        cmd.append("-O null")

    cmd.extend(["-o {}".format(file_name), orc_name, sco_name])
    if remove_files is True:
        cmd.append("; rm {}; rm {}".format(orc_name, sco_name))

    return subprocess.Popen(" ".join(cmd), shell=True)
