# Stable Diffusion + LoRA: archived team workflow

These five notebooks come from the supplied Stable Diffusion archive. The accompanying operating document attributes the workflow to teammate Yu-Hsin Tien (田宇新); Cheng-Jui Lai assisted with parameter testing and image generation, and handled the project's FID comparison work. This does not claim individual authorship of every cell.

## Order and required inputs

1. `1_lora_sam_drive.ipynb`: load a SAM ViT-H checkpoint, select center-point masks, crop and resize to 512 × 512. Provide JPG inputs and review the crops. The implementation is point-prompted segmentation, not an automatic crack detector; it does not itself generate captions.
2. `2_CLIP.ipynb`: use OpenCLIP ViT-L/14 with OpenAI weights to select the closest text from a supplied candidate list. Provide `prompt_candidates.txt` (not included). This ranks supplied captions; it does not generate free-form captions or train CLIP. Inspect the chosen captions manually.
3. `3_lora_training_colab_v2.ipynb`: train attention LoRA on Stable Diffusion v1.5 using matched images and same-stem TXT captions. Provide the image/caption ZIP. The saved weights stay in the notebook working directory unless you copy them to Drive.
4. `4_RealisticVision_v2_0_+_LoRA__CRACK.ipynb`: historical AUTOMATIC1111 setup, RealisticVision v2 checkpoint download and LoRA conversion. Model and LoRA weights are not included. Public Cloudflare tunnel launch cells have been removed. To use WebUI locally, use its upstream setup instructions and a local browser; a Colab runtime's localhost is not your laptop's localhost.
5. `Clean_FID.ipynb`: upload separate real/generated image ZIPs and compute clean and legacy_pytorch FID. Use a fresh runtime for a new pair: its extraction directories are reused and are not cleaned automatically. Upload only trusted ZIPs, with images at the archive root.

## Settings found in the supplied code/document

| Stage | Archived settings |
| --- | --- |
| LoRA training | resolution 512, batch 2, accumulation 2, epochs 16, learning rate 1e-5 |
| LoRA adapter | rank 8, alpha 32, dropout 0.05 |
| Generation (operating document) | DPM++ 3M SDE, Karras, 35 steps, CFG 7, 512 × 512 |
| Generation batch (operating document) | batch count 20, batch size 2: 40 images per run |
| Generation LoRA strength | 2.5, using `lora_crack_datasetV1_webui` |

Positive prompt recorded in the operating document:

```text
<lora:lora_crack_datasetV1_webui:2.5> close-up photo of a blue and white wind turbine blade, a single dark surface crack, visible fracture line, damaged surface, realistic black crack
```

Negative prompt:

```text
background crack, heavy fracture, large damage, unrealistic, cartoon, illustration, text, watermark
```

These are documented settings, not proof that every evaluated image used identical settings. Seeds and the complete generation history are unavailable.

## Limitations before rerunning

- Dependencies and external repositories were not pinned. Use separate environments for this workflow, YOLO and legacy pix2pixHD; do not assume current packages reproduce the old environment.
- The LoRA training loop only steps on complete accumulation groups. A remaining partial group is discarded at the end of each epoch. The historical algorithm is preserved, including this limitation.
- Mixed-precision training and PEFT/diffusers compatibility have not been tested here.
- The homemade PEFT-to-WebUI key conversion needs validation with the actual checkpoint. Correcting `return Non` to `return None` only fixes a typo; it does not establish valid conversion or alpha scaling. The exported file alone does not record all adapter configuration.
- The setup notebook mixes its own Python 3.10 environment with the Colab kernel. Installing a package in one does not guarantee availability in the other.
- SAM crop quality and CLIP caption choices require review. The operating document also mentions BLIP/CLIP Interrogator, but the supplied caption notebook implements candidate ranking with OpenCLIP.
- This package contains historical research code, not a verified one-click reproduction.

See `../../docs/INTEGRATION.md` for changes and FID evidence. External models and libraries retain their original authorship and terms.
