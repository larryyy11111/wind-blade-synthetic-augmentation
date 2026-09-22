# Recorded experiments

These settings were read from the supplied notebook, not inferred from the publication.

| Setting | Global experiment | Local experiment |
| --- | --- | --- |
| Experiment | exp_global | exp_local |
| Generator | global | local |
| ngf | 64 | 64 |
| label_nc | 10 | 10 |
| Batch size | 2 | 1 |
| niter / niter_decay | 100 / 50 | 30 / 20 |
| Learning rate | 0.0002 | 0.0001 |
| Instance maps | disabled | disabled |
| Local blocks | default | 3 |
| Pretrain argument | none | checkpoints/exp_global |

The saved global log reached epoch 150; the local log reached epoch 50. The local command requested continuation and a global pretrain path; these logs alone do not prove every intended weight was loaded. The notebook also records inference on three test labels. The archive does not include the trained checkpoint needed to rerun that inference.

The converter defines IDs 0–5, while the training command allocates 10 label classes. This discrepancy is retained and disclosed, not silently corrected. Match the class mapping and architecture to the intended checkpoint before reproducing a run.

FID/FID.py used pytorch-fid with 2048-dimensional features, batch size 32, cuda:0 and zero data-loader workers. Its folders held 500 real and 3 generated images. A prose note mentions 145, but no matched run metadata verifies it as a final result. No quantitative performance table is presented.

For future comparisons, preserve: dataset split; real and generated image counts; preprocessing and resolution; generation settings and seed; checkpoint; software versions; FID feature configuration; and detector evaluation protocol. FID evaluates distributional similarity, not defect localization accuracy. Do not use fixed score bands as universal quality labels.
