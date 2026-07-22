# DiffMSR PD/PD-FS Super-Resolution Ablations

This repository adapts the CVPR 2024 DiffMSR/McDiff code to proton-density knee MRI super-resolution with a proton-density fat-suppressed reference image.

Original paper: Guangyuan Li, Chen Rao, Juncheng Mo, Zhanjie Zhang, Wei Xing, Lei Zhao, "Rethinking Diffusion Model for Multi-Contrast MRI Super-Resolution", CVPR 2024.

## Data Mapping

DiffMSR's paired MRI loader expects MATLAB variables named `T1` and `T2`. For this project we keep those names but map them to our contrasts:

| DiffMSR field | Project data |
| --- | --- |
| `T2` | high-resolution PD target |
| `T2_128`, `T2_64` | k-space downsampled PD input |
| `T1`, `T1_128`, `T1_64` | PD fat-suppressed reference |

We used the Knee MRI dataset from the fastMRI bundle, second batch: https://fastmri.med.nyu.edu/

## Environment

```bash
bash scripts/create_diffmsr_env.sh
export PYTHON_BIN="$PWD/.conda/envs/diffmsr/bin/python"
export GPU_ID=3
export CPUS=4
"$PYTHON_BIN" scripts/check_diffmsr_env.py
```

## Prepare Dataset

```bash
export RAW_DB=/path/to/knee_mri_clinical_seq_batch2
export DATA_ROOT=mri_data_complex/mc_knee_pd_fs

OVERWRITE=1 bash scripts/wrap_prepare_pd_fs_dataset.sh "$RAW_DB" "$DATA_ROOT"
```

The preprocessing script discovers matched PD/PD-FS DICOM series, pairs slices by study/orientation, normalizes each slice, applies centered FFT, center-crops k-space to 256/128/64, applies inverse FFT, and writes DiffMSR `.mat` files.

Prepared dataset statistics:

| Property | Value |
| --- | ---: |
| paired studies | 1857 |
| train studies | 1486 |
| valid/test studies | 371 |
| train slices | 40372 |
| valid/test slices | 10119 |
| skipped slices | 54 |
| HR / mid / LR sizes | 256 / 128 / 64 |

The `valid/` split was held out from training and used as the reported test set. No third split was created.

## Exact Experiment YAMLs

Tracked YAMLs for the reported experiments are stored in:

```text
options/pd_fs_ablation/train/
options/pd_fs_ablation/test/
```

They assume:

```text
DATA_ROOT = mri_data_complex/mc_knee_pd_fs
checkpoints under experiments/
```

Run one exact training YAML:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/train.py \
  -opt options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  --launcher none
```

Run one exact test YAML:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/test.py \
  -opt options/pd_fs_ablation/test/latent_prior128_100k.yml \
  --launcher none
```

## Reproduce Experiments

### Experiment 1: Denoising Steps

This ablation changes Stage 2 diffusion denoising steps while reusing one shared Stage 1 checkpoint.

Shared Stage 1 used for the reported timestep ablation:

```text
experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```
Or from scratch, train a shared 256-dimensional Stage 1 once:

```bash
DATA_ROOT="$DATA_ROOT" STAGE=1 PRIOR_DIM=256 TOTAL_ITER=100000 \
  RUN_NAME=pd_fs_stage1_x4_prior256_100k RUN_TAG=prior256_stage1_100k \
  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_diffmsr_nohup.sh

export STAGE1_CKPT=experiments/pd_fs_stage1_x4_prior256_100k/models/net_g_latest.pth
```
Use the matching Stage 2 YAML and test YAML for the step count you want. For example, for 4 steps:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/train.py \
  -opt options/pd_fs_ablation/train/timesteps_steps4_stage2_100k.yml \
  --launcher none
```

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/test.py \
  -opt options/pd_fs_ablation/test/timesteps_steps4_100k.yml \
  --launcher none
```

The same pattern applies to the other step counts: `steps1`, `steps2`, `steps6`, and `steps8`.

Exact Stage 2 YAMLs:

```text
options/pd_fs_ablation/train/timesteps_steps1_stage2_100k.yml
options/pd_fs_ablation/train/timesteps_steps2_stage2_100k.yml
options/pd_fs_ablation/train/timesteps_steps4_stage2_100k.yml
options/pd_fs_ablation/train/timesteps_steps6_stage2_100k.yml
options/pd_fs_ablation/train/timesteps_steps8_stage2_100k.yml
```

Matching test YAMLs:

```text
options/pd_fs_ablation/test/timesteps_steps1_100k.yml
options/pd_fs_ablation/test/timesteps_steps2_100k.yml
options/pd_fs_ablation/test/timesteps_steps4_100k.yml
options/pd_fs_ablation/test/timesteps_steps6_100k.yml
options/pd_fs_ablation/test/timesteps_steps8_100k.yml
```

Wrapper command to regenerate/run the same experiment family:

```bash
export STAGE1_CKPT=experiments/pd_fs_stage1_x4/models/net_g_latest.pth

for steps in 1 2 4 6 8; do
  STAGE1_CKPT="$STAGE1_CKPT" \
  bash scripts/wrap_train_denoising_steps.sh "$DATA_ROOT" 100000 "$steps" "$GPU_ID"
done
```

### Experiment 2: Latent Vector Size

This ablation changes the 1D diffusion prior/latent vector length. Because `prior_dim` changes the Stage 1 latent interface, each setting trains a matching Stage 1 and Stage 2 pair.

Train and test each pair directly with the matching YAMLs. For example, for 128-dimensional latent prior:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/train.py \
  -opt options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  --launcher none
```

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/train.py \
  -opt options/pd_fs_ablation/train/latent_prior128_stage2_100k.yml \
  --launcher none
```

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/test.py \
  -opt options/pd_fs_ablation/test/latent_prior128_100k.yml \
  --launcher none
```

The same pattern also works for `64`, `256`, and `512` by swapping the YAML filenames.

Exact YAML pairs:

```text
options/pd_fs_ablation/train/latent_prior64_stage1_100k.yml  -> options/pd_fs_ablation/train/latent_prior64_stage2_100k.yml
options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml -> options/pd_fs_ablation/train/latent_prior128_stage2_100k.yml
options/pd_fs_ablation/train/latent_prior256_stage1_100k.yml -> options/pd_fs_ablation/train/latent_prior256_stage2_100k.yml
options/pd_fs_ablation/train/latent_prior512_stage1_100k.yml -> options/pd_fs_ablation/train/latent_prior512_stage2_100k.yml
```

Wrapper command:

```bash
for dim in 64 128 256 512; do
  bash scripts/wrap_train_latent_size.sh "$DATA_ROOT" 100000 "$dim" "$GPU_ID"
done
```

### Experiment 3: Latent-Space vs Image-Space Diffusion

This section is reserved for the partner experiment comparing DiffMSR's latent-space diffusion prior against an image-space diffusion variant.

| Variant | Training YAML | Test YAML | PSNR | SSIM |
| --- | --- | --- | ---: | ---: |
| latent-space diffusion | `baseline_stage1_prior256_500k.yml` + `baseline_stage2_prior256_500k.yml` | `baseline_prior256_500k.yml` | 30.5910 | 0.8473 |
| image-space diffusion | `train_S1_image.yml` + `train_S2_image.yml` | `test_image.yml` | 30.3210 | 0.8413 |

## Test And Figures

Run full held-out tests:

```bash
DATA_ROOT="$DATA_ROOT" TEST_GROUP=timesteps GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_pd_fs_full_tests.sh

DATA_ROOT="$DATA_ROOT" TEST_GROUP=latent GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_pd_fs_full_tests.sh
```

Useful `TEST_GROUP` values: `baseline`, `timesteps`, `latent`, `latent256`, `all`.

Generate metric plots, side-by-side examples, and compute-time plots:

```bash
"$PYTHON_BIN" scripts/visualize_pd_fs_ablation_results.py --top-n 5 --middle-slice-fraction 0.6
"$PYTHON_BIN" scripts/plot_pd_fs_compute_time.py
```

Outputs:

```text
analysis_outputs/pd_fs_ablation/metrics_summary.csv
analysis_outputs/pd_fs_ablation/metrics_latent.png
analysis_outputs/pd_fs_ablation/metrics_timesteps.png
analysis_outputs/pd_fs_ablation/latent_examples/
analysis_outputs/pd_fs_ablation/timestep_examples/
analysis_outputs/pd_fs_ablation/compute_time_summary.csv
analysis_outputs/pd_fs_ablation/compute_time_latent.png
analysis_outputs/pd_fs_ablation/compute_time_timesteps.png
```
To generate the latent-vs-image-space comparison figures, first run the image-space model test so it creates a DiffMSR-style result folder:

```text
results/pd_fs_test_image_space/visualization/*.mat
```

The `.mat` file must contain `recon` and `gt` arrays in the same validation-slice order as `mri_data_complex/mc_knee_pd_fs/valid/`. Then run:

```bash
"$PYTHON_BIN" scripts/visualize_pd_fs_ablation_results.py \
  --experiments space \
  --space-models latent=pd_fs_test_baseline_prior256_500k,image=pd_fs_test_image_space \
  --top-n 5 \
  --middle-slice-fraction 0.6
```

This writes:

```text
analysis_outputs/pd_fs_ablation/metrics_space.png
analysis_outputs/pd_fs_ablation/space_examples/
analysis_outputs/pd_fs_ablation/selected_space_examples.csv
```

If her run name is different, replace `pd_fs_test_image_space` with the folder name under `results/`.

## Results

All metrics are on the held-out `valid/` split.

### Denoising Steps, 100k Stage 2

| steps | PSNR | SSIM | Stage 2 + test time |
| ---: | ---: | ---: | ---: |
| 1 | 29.9231 | 0.8349 | 4.62 h |
| 2 | 30.1042 | 0.8372 | 4.74 h |
| 4 | 30.1055 | 0.8373 | 6.22 h |
| 6 | 30.1034 | 0.8372 | 4.43 h |
| 8 | 30.1064 | 0.8373 | 4.54 h |

Best PSNR was 8 steps. Four steps was essentially tied and is the original setting.

### Latent Vector Size, 100k Stage 1 + 100k Stage 2

| `prior_dim` | PSNR | SSIM | total time |
| ---: | ---: | ---: | ---: |
| 64 | 30.0977 | 0.8371 | 8.96 h |
| 128 | 30.1032 | 0.8372 | 9.20 h |
| 256 | 30.0834 | 0.8368 | 5.04 h |
| 512 | 30.0763 | 0.8367 | 9.51 h |

The best 100k latent-size result was `prior_dim=128`, but differences among 64/128/256 were small.

### Image Space, 500k Stage 1 + 500k Stage 2

| `prior_dim` | PSNR | SSIM |
| ---: | ---: | ---: |
| 256 | 30.3210 | 0.8413 |

The result was not as good as the latent space baseline. That supports our conjecture that the latent space, other than being more efficient, also produces better results.

### Full Baseline

| model | PSNR | SSIM |
| --- | ---: | ---: |
| 256-dimensional latent, full Stage-2 run | 30.5910 | 0.8473 |

## Smoke Tests

Fast smoke test, no real training:

```bash
RUN_PREPROCESS_SMOKE=0 bash scripts/smoke_test_pd_fs_wrappers.sh
```

Smoke test including a tiny DICOM conversion:

```bash
RAW_DB=/path/to/extracted_dicom_folder RUN_PREPROCESS_SMOKE=1 \
  bash scripts/smoke_test_pd_fs_wrappers.sh
```

Use an already extracted DICOM folder when possible. If `RAW_DB` is an archive, the smoke script skips extraction by default; set `ALLOW_ARCHIVE_EXTRACT=1` only if full archive extraction is intentional.

## General Training Utilities

Use the wrappers to create new variants without hand-editing YAML:

```bash
# Stage 1 or Stage 2 from template YAMLs
DATA_ROOT="$DATA_ROOT" STAGE=1 TOTAL_ITER=100000 PRIOR_DIM=256 \
  RUN_NAME=my_stage1 RUN_TAG=my_stage1 bash scripts/run_diffmsr_nohup.sh

DATA_ROOT="$DATA_ROOT" STAGE=2 TOTAL_ITER=100000 TIMESTEPS=4 PRIOR_DIM=256 \
  PRETRAIN_S1=experiments/my_stage1/models/net_g_latest.pth \
  RUN_NAME=my_stage2 RUN_TAG=my_stage2 bash scripts/run_diffmsr_nohup.sh
```

Generated options are written to `logs/generated_options/`. Training outputs are written to `experiments/<run_name>/`.

## Notes

- Raw DICOMs, converted `.mat` files, checkpoints, logs, result images, and analysis outputs are ignored by git.
- The degradation follows the original DiffMSR demo style: center-cropped k-space and inverse FFT. It is a controlled super-resolution degradation, not a realistic accelerated MRI sampling mask.
- Test YAMLs include an inherited `train:` block from the original template, but `DiffMSR_Main/test.py` only uses the dataset, network, and checkpoint fields.

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [SRFormer](https://github.com/HVision-NKU/SRFormer).
