# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import os
import shutil
import argparse


def copy_images(src_folder, dest_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through all subdirectories and files in the source folder
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # Check if the file has an image extension (e.g., .jpg, .png, .jpeg, etc.)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg")):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)

                # Copy the image file to the destination folder
                shutil.copy(src_file, dest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("copy.py")
    parser.add_argument("-s", "--source", help="Path to source folder", required=True)
    parser.add_argument(
        "-d", "--dest", help="Path to destination folder", required=True
    )

    args = parser.parse_args()

    source_folder = args.source
    destination_folder = args.dest

    copy_images(source_folder, destination_folder)
