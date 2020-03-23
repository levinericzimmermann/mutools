"""This module contains small functions for using csound within Python"""

import subprocess


def render_csound(file_name: str, orc_name: str, sco_name: str) -> None:
    cmd = (
        "csound",
        "-d",
        "--format=double",
        "-k 96000",
        "-r 96000",
        "-o {0}".format(file_name),
        orc_name,
        sco_name,
    )
    subprocess.call(" ".join(cmd), shell=True)
