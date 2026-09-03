# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

import json
import os
import tqdm


def coco_to_yolo(coco_json_path: str, output_dir: str, img_dir: str):
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"Path not found: '{coco_json_path}'")
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    category_id_to_name = {
        category["id"]: category["name"] for category in coco_data["categories"]
    }
    category_name_to_id = {
        name: i for i, (id, name) in enumerate(category_id_to_name.items())
    }

    for image in tqdm.tqdm(coco_data["images"]):
        img_id = image["id"]
        img_filename = image["file_name"]

        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        if (
            not os.path.exists(os.path.dirname(txt_path))
            and os.path.dirname(txt_path) != ""
        ):
            raise FileNotFoundError(f"Directory for '{txt_path}' does not exist.")
        with open(txt_path, "w") as txt_file:
            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == img_id:
                    category_id = annotation["category_id"]
                    category_name = category_id_to_name[category_id]
                    yolo_class = category_name_to_id[category_name]

                    bbox = annotation["bbox"]
                    x, y, width, height = bbox

                    # Convert to YOLO format
                    x_center = x + width / 2
                    y_center = y + height / 2
                    x_center /= image["width"]
                    y_center /= image["height"]
                    width /= image["width"]
                    height /= image["height"]

                    txt_file.write(
                        f"{yolo_class} {x_center} {y_center} {width} {height}\n"
                    )


if __name__ == "__main__":
    coco_json_path_train = "annotations/instances_train2017.json"
    coco_json_path_val = "annotations/instances_val2017.json"
    output_dir_train = "labels/train"
    output_dir_val = "labels/val"
    img_dir_train = "images/train"
    img_dir_val = "images/val"

    coco_to_yolo(coco_json_path_train, output_dir_train, img_dir_train)
    coco_to_yolo(coco_json_path_val, output_dir_val, img_dir_val)
