# Wind Turbine Blade Defect Detection: Synthetic Image Experiments

A research portfolio by **Cheng-Jui Lai**, presenting the semantic image synthesis and FID evaluation work from a team capstone on wind turbine blade defect detection.

**Scope:** this repository contains the pix2pixHD experiment code and evaluation utilities recovered from the project archive. The broader team project also used Stable Diffusion and YOLOv9; those implementations are not included in this package.

[Related publication on IEEE Xplore](https://ieeexplore.ieee.org/document/11661275)

## Research motivation

The project explored synthetic image augmentation for wind turbine blade defect detection with limited labeled data. Semantic label maps were used to guide image generation, and Fréchet Inception Distance (FID) was used to compare generated and real image distributions.

## My contribution

- Conducted the semantic image synthesis experiments represented by this pix2pixHD archive.
- Performed FID evaluation of generated images.
- Assisted with Stable Diffusion parameter adjustments and image generation.

My teammates primarily handled Stable Diffusion and YOLOv9. Their work is credited as part of the team project, not claimed as my individual implementation. The underlying pix2pixHD model is third-party research software, not an architecture I developed.

## Naming clarification

The source archive was named `GauGAN.zip`, but its README, model code, and training notebook identify **NVIDIA pix2pixHD**. This repository uses the implementation name supported by those files. The archive name should not be treated as proof that a SPADE/GauGAN model was trained.

## Repository contents

| Path | Purpose |
| --- | --- |
| `third_party/pix2pixHD/` | Python source recovered from the uploaded archive, with original license |
| `notebooks/original_training_commands.ipynb` | Historical Colab commands; outputs and metadata cleared |
| `scripts/convert_labelme.py` | LabelMe JSON to class-ID masks, adapted to accept paths |
| `scripts/resize_images.py` | Resize images or masks while preserving source files |
| `scripts/evaluate_fid.py` | Configurable FID evaluation with optional JSON results |
| `docs/EXPERIMENTS.md` | Recorded settings and evidence limitations |
| `docs/SETUP.md` | Setup and training instructions |
| `docs/PROVENANCE.md` | Source attribution and packaging changes |

## Quick start: FID

Use Python 3 in an isolated environment. Install PyTorch and torchvision for your platform, then install `requirements.txt`. See [setup instructions](docs/SETUP.md).

```bash
python scripts/evaluate_fid.py --real data/fid_real --fake data/fid_fake --device cpu --output outputs/fid_run.json
```

For a configured CUDA environment, use `--device cuda:0`. The first evaluation may download Inception weights. Images are read directly from each folder, not recursively. No datasets or model weights are bundled.

The wrapper records sample counts and settings. It does not reproduce a published result by itself: that requires the original image sets, preprocessing, model checkpoint, and environment.

## Data preparation

```bash
python scripts/convert_labelme.py --input data/annotations --output data/raw_masks
python scripts/resize_images.py --input data/raw_masks --output data/train_label --size 512 --kind mask
python scripts/resize_images.py --input data/images --output data/train_img --size 512 --kind image
```

The size above is an example, not a verified historical configuration. Pair each image and mask by filename stem. Masks use nearest-neighbor interpolation and lossless PNG output. The converter retains the supplied class mapping: background/sky=0, blade=1, tower/pillar=2, crack=3, floor/ground=4, fence=5. Unknown labels stop conversion for explicit correction. Existing masks with the same name are overwritten by the converter; choose a fresh output folder. Class-presence checks do not establish pixel-level annotation accuracy.

## Results and limitations

The original notebook contains completed training logs, but this package does not claim a verified FID score or detector improvement. The archived FID folders held 500 real images and only 3 generated images, which is not an adequate basis for presenting a stable final comparison. A note mentioning 145 was not accompanied by a matching evaluation log.

Compare FID under a consistent protocol; no universal “good/bad” threshold is claimed here. See [experiment notes](docs/EXPERIMENTS.md). End-to-end training and FID were not rerun during packaging; the historical model code may require compatibility updates for modern PyTorch.

## Attribution and reuse

pix2pixHD: Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro, *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs*, CVPR 2018. Upstream: https://github.com/NVIDIA/pix2pixHD

The original [third-party license](third_party/pix2pixHD/LICENSE.txt) is preserved. No new blanket license has been assigned to the team's own code. Dataset images, annotations, model weights, and the publisher PDF are excluded pending confirmation of redistribution scope. This is a review package, not a claim that all team materials are cleared for public release.
