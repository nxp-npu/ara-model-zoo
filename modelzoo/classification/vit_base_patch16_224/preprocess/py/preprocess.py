# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import numpy
import copy

from core.python.config import Config
from core.python.preprocess.preprocessing_builder import PreprocessingBuilder
from core.python.preprocess.operations.bgrtorgb import BgrToRgb
from core.python.preprocess.interfaces import PreprocessOutput

from PIL import Image
from torchvision import transforms

# This preprocess function is specific to ViT Base Patch16 224 model and is designed to be used in the preprocess.py script for that model. It applies BGR to RGB conversion and resizing
# to 224x224 using bicubic interpolation, which are the expected preprocessing steps for this model. 
# The function takes an input image as a numpy array and a Config object, and returns a PreprocessOutput object containing the preprocessed image ready for inference.
def preprocess(image: numpy.ndarray, config: Config) -> PreprocessOutput:
    # This deep copy is necessary otherwise original config will be modified
    # and when we would run inference 2nd time then we would not have value of resize_size
    updated_config: Config = copy.deepcopy(config)

    # Modify the updated_config accordingly
    updated_config.preprocess.bgr_to_rgb = False
    resize_size = (
        updated_config.preprocess.resize.resize_size
        if (
            updated_config.preprocess.resize is not None
            and updated_config.preprocess.resize.resize_size is not None
        )
        else None
    )
    updated_config.preprocess.resize = None

    # Apply BGR to RGB operation
    bgr_to_rgb = BgrToRgb()
    image = bgr_to_rgb(image)

    # Appy resize operation
    transform = transforms.Resize(
        resize_size, interpolation=transforms.InterpolationMode.BICUBIC
    )
    img_pil = Image.fromarray(image)
    img_pil_resized = transform(img_pil)
    image_resized = numpy.array(img_pil_resized)

    # Build the remaining preprocessing pipeline and execute it
    return PreprocessingBuilder(updated_config).execute_pipeline(image_resized)
