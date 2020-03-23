"""This module contains small functions for using csound within Python"""

import os


def render_csound(file_name: str, orc_name: str, sco_name: str) -> None:
    cmd0 = "csound --format=double -k 96000 -r 96000 -o {0} ".format(file_name)
    cmd1 = "{0} {1}".format(orc_name, sco_name)
    cmd = cmd0 + cmd1
    os.system(cmd)
