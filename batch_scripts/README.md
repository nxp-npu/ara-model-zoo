# Ara-model-zoo Scripts

> **Note:** All batch scripts must be run from the repository **root** directory.
> Running them from inside `batch_scripts/` (or any other directory) will fail
> because they rely on relative paths to the project layout. The examples below
> therefore invoke the scripts as `python batch_scripts/<script>.py`.

## Setup

### models.txt

Create a models file with one model name per line (`#` for comments):

```
yolov8n
resnet50v1
# face_det   <- skipped
```

### eval_config.json

Required for evaluation. Maps model names to their dataset paths:

```json
{
    "yolov8n":  { "dataset_root": "/datasets/coco2017" },
    "resnet50": { "dataset_root": "/datasets/imagenet" }
}
```

## Compile

Exactly one of `--dataset-root` (a single dataset for the whole batch) or `--dataset-config`
(a JSON file mapping each model to its dataset) is required.

```bash
python batch_scripts/batch_compile.py --models yolov8n --dataset-root /datasets/coco2017
python batch_scripts/batch_compile.py --file batch_scripts/models.txt --dataset-config eval_config.json
python batch_scripts/batch_compile.py --file batch_scripts/models.txt --dataset-root /datasets/coco2017 --cleanup   # auto-delete extra output folders
python batch_scripts/batch_compile.py --file batch_scripts/models.txt --dataset-root /datasets/coco2017 --dry-run   # preview only
```

## Evaluate

```bash
python batch_scripts/batch_eval_hw.py                                      # uses eval_config.json
python batch_scripts/batch_eval_hw.py --config custom_config.json
python batch_scripts/batch_eval_hw.py --dry-run
```

## Benchmark

```bash
python batch_scripts/batch_performance.py --file batch_scripts/models.txt
python batch_scripts/batch_performance.py --models yolov8n resnet50
python batch_scripts/batch_performance.py --file batch_scripts/models.txt --dry-run
```

## Cleanup

```bash
python batch_scripts/cleanup_output.py                                     # delete extra output folders
python batch_scripts/cleanup_output.py --dry-run                           # preview what would be deleted
python batch_scripts/cleanup_output.py --task object_detection --model yolov8n
```

## Logs

Logs are written to `logs/<timestamp>/<script>/`. Override with `--log-dir`.

```
logs/
└── 20250204_143025/
    ├── compile/
    ├── eval_hw/
    └── performance/
```
