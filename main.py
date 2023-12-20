#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file, entry point
"""
__author__ = "AsifulHasan"
__copyright__ = "Copyright 2023, "
__date__ = "2023/11/23"
__version__ = "0.0.1"


import sys
from streamlit import runtime
from streamlit.web import cli as stcli
from tool import tool


def run():
    if runtime.exists():
        tool()
    else:
        sys.argv = ["streamlit", "run", __file__] + sys.argv
        sys.exit(stcli.main())


if __name__ == '__main__':
    run()