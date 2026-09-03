# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add sigmoid heads to SCRFD classification outputs and extract a pruned "
            "model with a consistent cls/bbox/kps output ordering."
        )
    )
    parser.add_argument(
        "--input-model", required=True, help="Path to the input ONNX model."
    )
    parser.add_argument(
        "--output-model", required=True, help="Path to write the pruned ONNX model."
    )
    parser.add_argument(
        "--cls-nodes",
        nargs=3,
        required=True,
        metavar=("STRIDE8", "STRIDE16", "STRIDE32"),
        help="Raw classification node names for strides 8, 16, and 32.",
    )
    parser.add_argument(
        "--bbox-nodes",
        nargs=3,
        required=True,
        metavar=("STRIDE8", "STRIDE16", "STRIDE32"),
        help="Raw bounding-box node names for strides 8, 16, and 32.",
    )
    parser.add_argument(
        "--kps-nodes",
        nargs=3,
        required=True,
        metavar=("STRIDE8", "STRIDE16", "STRIDE32"),
        help="Raw keypoint node names for strides 8, 16, and 32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_model = Path(args.input_model)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_model))

    # New onnxsim fuses box-scale Mul into Conv, so 441/461/481 disappear.
    # The fused Conv keeps the previous tensor name.
    existing = {output for node in model.graph.node for output in node.output}
    for bbox_name in args.bbox_nodes:
        if bbox_name in existing:
            continue
        fused = str(int(bbox_name) - 1)
        if fused not in existing:
            raise RuntimeError(
                f"Bounding-box tensor {bbox_name!r} is missing after simplify "
                f"and fused source {fused!r} was not found"
            )
        model.graph.node.append(
            helper.make_node(
                "Identity",
                inputs=[fused],
                outputs=[bbox_name],
                name=f"Restore_{bbox_name}",
            )
        )

    sigmoid_output_names: list[str] = []
    sigmoid_nodes = []
    for cls_node in args.cls_nodes:
        sigmoid_name = f"{cls_node}_sigmoid"
        sigmoid_output_names.append(sigmoid_name)
        sigmoid_nodes.append(
            helper.make_node(
                "Sigmoid",
                inputs=[cls_node],
                outputs=[sigmoid_name],
                name=f"Sigmoid_{cls_node}",
            )
        )

    model.graph.node.extend(sigmoid_nodes)

    temp_model = output_model.with_name(f"{output_model.stem}.with_sigmoid.onnx")
    onnx.save(model, str(temp_model))

    output_names: list[str] = []
    for cls_name, bbox_name, keypoint_name in zip(
        sigmoid_output_names,
        args.bbox_nodes,
        args.kps_nodes,
        strict=True,
    ):
        output_names.extend([cls_name, bbox_name, keypoint_name])

    utils.extract_model(
        str(temp_model),
        str(output_model),
        input_names=[value.name for value in model.graph.input],
        output_names=output_names,
    )

    temp_model.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
