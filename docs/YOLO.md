# YOLO detector code and dataset configurations

The supplied `yolo.zip` contains an Ultralytics source tree whose version string is **8.3.67**, project dataset YAML files, and general upstream examples. It does not include a project-specific training script, run arguments, detector logs, split manifests or the team's trained detector weights. Its exact upstream commit and local modifications have not been established; the version string alone is not an environment lockfile.

`third_party/ultralytics/` preserves the runtime Python modules, model/configuration YAML files, small package assets, package metadata, upstream README/citation and LICENSE. It supports multiple detector families. It is third-party code, not a student-authored detector architecture. Upstream CI, editor caches, generic documentation/examples/tests, Docker files and the bundled PT checkpoint were excluded.

## Setup

Use an isolated environment with a compatible PyTorch/torchvision installation:

```bash
python -m pip install -e third_party/ultralytics
```

This installs the archived source, including its declared dependencies. It was not installed or GPU-tested during packaging.

## Prepare your data

Edit `path:` in the desired YAML under `third_party/ultralytics/ultralytics/cfg/datasets/`. The original Windows account paths have been replaced with explicit placeholders. All 23 supplied dataset configurations define one class: `0: crack`, with `train/images`, `valid/images`, and `test/images` splits. Matching YOLO TXT annotations belong in each split's `labels` directory.

Dataset names (`r150`, `g150`, `50_3000`, etc.) are retained from the archive. Verify the meaning and actual membership of each dataset rather than inferring a real/synthetic ratio from a filename. The report's results are not mapped to verified individual YAML/run pairs.

## Example commands — newly documented, not recovered experiment commands

After editing the dataset path, one illustrative training invocation is:

```bash
yolo detect train model=yolov9c.pt data=third_party/ultralytics/ultralytics/cfg/datasets/50_3000.yaml epochs=200 imgsz=512 batch=8 project=outputs/yolo name=example_50_3000
```

The model variant, epochs and batch above are example choices; they do not establish the exact run behind a report table. The requested pretrained checkpoint may be downloaded on first use. To evaluate your resulting checkpoint on the held-out test split:

```bash
yolo detect val model=outputs/yolo/example_50_3000/weights/best.pt data=third_party/ultralytics/ultralytics/cfg/datasets/50_3000.yaml split=test imgsz=512
```

Record the actual software versions, seed, checkpoint, training settings and image-level split lists before comparisons. Keep held-out detector images separate from both detector training and generator training.

The upstream license is retained at `../third_party/ultralytics/LICENSE`. No blanket license is assigned to the entire capstone repository.
