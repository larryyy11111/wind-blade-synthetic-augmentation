# Setup and reproduction notes

## Evaluation and preprocessing

Create and activate a Python 3 virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`.
Linux/macOS: `source .venv/bin/activate`.

Install a mutually compatible PyTorch/torchvision build for your system, then:

```bash
python -m pip install -r requirements.txt
```

This dependency list is not version-pinned because the original environment was not recovered. No current GPU environment was validated during packaging. FID requires Inception weights (first use may require network access), and pix2pixHD training may download VGG weights.

## Historical model training

From the project root, enter the model directory:

```bash
cd third_party/pix2pixHD
```

Supply your authorized training data under `datasets/train_img1/train_label` and `datasets/train_img1/train_img`. Each grayscale class-ID label must match an RGB photograph by filename stem. Do not put color visualizations in the training label folder.

Historical global command:

```bash
python train.py --name exp_global --netG global --ngf 64 --dataroot ./datasets/train_img1 --label_nc 10 --no_instance --batchSize 2 --nThreads 0 --gpu_ids 0 --niter 100 --niter_decay 50 --lr 0.0002
```

Historical local command (requires the global checkpoint):

```bash
python train.py --name exp_local --netG local --ngf 64 --dataroot ./datasets/train_img1 --label_nc 10 --no_instance --batchSize 1 --n_blocks_local 3 --nThreads 0 --gpu_ids 0 --niter 30 --niter_decay 20 --lr 0.0001 --continue_train --load_pretrain checkpoints/exp_global
```

These are recovered commands, not verified instructions for a modern PyTorch installation. Inspect checkpoint-loading messages and dataset settings before relying on the local continuation. The original notebook uses Colab and Google Drive; change its working directory for your own environment. Its historical test command should be adapted for the host shell. Prefer an explicit `--results_dir ./results`.

Training may have additional dependencies associated with optional features such as TensorRT, ONNX, or apex. Those optional execution paths are not covered by the basic requirements file. Original source was retained rather than modernized without GPU validation.
