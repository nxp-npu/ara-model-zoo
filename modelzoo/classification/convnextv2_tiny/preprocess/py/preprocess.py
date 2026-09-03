# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

"""
Preprocess input image for model inference using torchvision.

Steps:
- Convert BGR to RGB
- Resize and center crop to 224x224
- Convert to tensor
- Normalize using ImageNet statistics

Args:
    image (numpy.ndarray): Input image in BGR format.
    config (Config): Configuration object (not used currently).

Returns:
    PreprocessOutput: Original PIL image and processed output array/tensor.
"""
import numpy
from core.python.config import Config
from core.python.preprocess.operations.bgrtorgb import BgrToRgb
from core.python.preprocess.interfaces import PreprocessOutput

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def preprocess(image: numpy.ndarray, config: Config) -> PreprocessOutput:
    # Apply BGR to RGB operation
    bgr_to_rgb = BgrToRgb()
    image = bgr_to_rgb(image)

    # Define transform operations
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    
    img_pil = Image.fromarray(image)

    # Apply transforms
    img_pil_transformed = transform(img_pil)

    img_pil_transformed = numpy.array(img_pil_transformed)

    return PreprocessOutput(img_pil, img_pil_transformed)

