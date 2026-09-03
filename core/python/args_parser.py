# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="path to data YAML configuration file"
    )
    parser.add_argument(
        "--mode",
        default="quick",
        required=False,
        choices=["quick", "normal", "deploy", "prepackaged"],
        help="type of run and the corresponding parameters to pick from YAML configuration file",
    )
    parser.add_argument(
        "--accuracy-target",
        default="ara",
        required=False,
        choices=["ara", "dvnc"],
        help="output files to read based on the mode i.e. Ara-1 output or compiler output",
    )
    parser.add_argument(
        "--name", default="", required=False, help="name of configuration"
    )
    return parser.parse_args()
