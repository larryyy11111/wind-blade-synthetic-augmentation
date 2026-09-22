# Provenance and packaging changes

Source: user-supplied GauGAN.zip. This package is a curated copy; the original archive is unchanged.

- third_party/pix2pixHD: Python source preserved (text encoding/newline normalization only), excluding the ad-hoc correct.py and make_placeholders.py helpers. Original LICENSE.txt retained. This is an archived derivative; it has not been diffed against a known upstream commit. Do not describe every included source line as original student work.
- scripts/convert_labelme.py: adapted from the archive's batch_convert and check label mask match.py. Replaced local paths with CLI arguments, rejects unknown labels, and uses decoded JSON image data consistently. Original class mapping is preserved.
- scripts/evaluate_fid.py: new portable wrapper based on the archive's FID/FID.py. Keeps 2048-dimensional features and zero workers; adds folder validation, sample counts, small-sample warning and JSON output. Default device is CPU; original script defaulted to CUDA.
- scripts/resize_images.py: newly written packaging utility based on the original resizing task. Uses explicit image/mask selection, separate output folders and collision checks.
- notebooks/original_training_commands.ipynb: original commands preserved, saved output and notebook metadata cleared. Historical completion evidence was inspected before clearing and summarized in EXPERIMENTS.md.
- README and documentation: newly prepared from archive contents and the user's stated contribution. These documentation/packaging changes are not presented as work performed during the original research.

Excluded: dataset images and embedded-image annotations, generated images, upstream demo media, caches, repository history, temporary files, the heuristic FID score note, and redundant path-specific helpers. Stable Diffusion and YOLOv9 archives have not been incorporated into this package.

Personal contribution was confirmed by the user: semantic image generation experiments, FID evaluation, and assistance with Stable Diffusion parameter tuning and image generation. Authorship of each auxiliary preprocessing script was not independently established.
