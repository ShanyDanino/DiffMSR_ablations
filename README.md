# DiffMSR PD/PD-FS Super-Resolution Ablations

This repository is based on:

Guangyuan Li, Chen Rao, Juncheng Mo, Zhanjie Zhang, Wei Xing, Lei Zhao, "Rethinking Diffusion Model for Multi-Contrast MRI Super-Resolution", CVPR 2024.

The original project trains multi-contrast MRI super-resolution with a low-resolution target contrast and a high-resolution reference contrast. In this fork we use:

| DiffMSR field | Our data |
| --- | --- |
| `T2` | high-resolution proton-density MRI |
| `T2_128`, `T2_64` | k-space downsampled proton-density MRI |
| `T1`, `T1_128`, `T1_64` | proton-density fat-suppressed reference MRI |

The `T1`/`T2` names are kept because parts of the original DiffMSR data loader expect these MATLAB variable names.

## Environment

Create the local environment and point the wrappers at its Python:

```bash
bash scripts/create_diffmsr_env.sh
export PYTHON_BIN="$PWD/.conda/envs/diffmsr/bin/python"
"$PYTHON_BIN" scripts/check_diffmsr_env.py
```

On the Technion server we usually run on one visible GPU:

```bash
export GPU_ID=3
export CPUS=4
```

## 1. Prepare The Dataset

We used the Knee MRI dataset that is a part of the fast MRI bundle (second batch). The database is available to download here: https://fastmri.med.nyu.edu/

The preparation wrapper accepts either an extracted DICOM database folder or a supported archive.

```bash
export RAW_DB=/path/to/knee_mri_clinical_seq_batch2
export DATA_ROOT=mri_data_complex/mc_knee_pd_fs

OVERWRITE=1 bash scripts/wrap_prepare_pd_fs_dataset.sh "$RAW_DB" "$DATA_ROOT"
```

This calls `scripts/preprocess_pd_fs_dicom.py`, which:

- discovers PD and PD fat-suppressed DICOM series per study and orientation;
- pairs PD slices with the matching fat-suppressed reference slices;
- normalizes each slice, applies centered FFT, center-crops k-space to 256, 128, and 64, then applies inverse FFT;
- writes DiffMSR `.mat` files with `T2`, `T2_128`, `T2_64`, `T1`, `T1_128`, and `T1_64`;
- performs a study-level split into `train/` and `valid/`, and copies `complex_data_demo/dc_mask/lr_4x.mat` to `dc_mask.mat`.

For our processed database:

| Property | Value |
| --- | ---: |
| paired studies | 1857 |
| train studies | 1486 |
| valid studies | 371 |
| train slices | 40372 |
| valid/test slices | 10119 |
| skipped slices | 54 |
| HR / mid / LR sizes | 256 / 128 / 64 |

We did not create a separate third test split. The `valid/` split was held out from training and used as the final reported test set.

## 2. Train Denoising-Step Ablations

Changing the number of diffusion denoising steps only changes Stage 2. For a clean ablation, first train or choose one shared Stage-1 checkpoint, then reuse it across all denoising-step runs.

Exact Stage-1 checkpoint used for our reported timestep ablation:

```bash
export STAGE1_CKPT=experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```

From scratch, train a shared 256-dimensional Stage 1 once:

```bash
DATA_ROOT="$DATA_ROOT" STAGE=1 PRIOR_DIM=256 TOTAL_ITER=100000 \
  RUN_NAME=pd_fs_stage1_x4_prior256_100k RUN_TAG=prior256_stage1_100k \
  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_diffmsr_nohup.sh

export STAGE1_CKPT=experiments/pd_fs_stage1_x4_prior256_100k/models/net_g_latest.pth
```

Run one denoising-step Stage-2 setting (for example, the original setting of 4 denoising steps and with 100000 iterations):

```bash
bash scripts/wrap_train_denoising_steps.sh "$DATA_ROOT" 100000 4 "$GPU_ID"
```

Run the full set sequentially with `nohup`:

```bash
mkdir -p logs
nohup bash -lc '
set -e
for steps in 1 2 4 6 8; do
  STAGE1_CKPT="$STAGE1_CKPT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/wrap_train_denoising_steps.sh "$DATA_ROOT" 100000 "$steps" "$GPU_ID"
done
' > logs/nohup_timesteps_100k.log 2>&1 &
```

Alternatively, for a quick single-command smoke run, the wrapper can train Stage 1 if the checkpoint is missing:

```bash
AUTO_TRAIN_STAGE1=1 bash scripts/wrap_train_denoising_steps.sh "$DATA_ROOT" 100000 8 "$GPU_ID"
```

For the full denoising-step ablation, prefer the explicit shared Stage-1 command above so all Stage-2 runs truly use the same Stage-1 weights.

## 3. Train Latent-Size Ablations

Changing `prior_dim` changes the Stage-1 latent interface, so each latent-size experiment trains Stage 1 and then Stage 2 with the matching checkpoint.

Run one latent-size setting (for example, the original setting of 256 latent vector size and with 100000 iterations):

```bash
bash scripts/wrap_train_latent_size.sh "$DATA_ROOT" 100000 256 "$GPU_ID"
```

Run all latent-size settings sequentially with `nohup`:

```bash
mkdir -p logs
nohup bash -lc '
set -e
for dim in 64 128 256 512; do
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/wrap_train_latent_size.sh "$DATA_ROOT" 100000 "$dim" "$GPU_ID"
done
' > logs/nohup_latent_100k.log 2>&1 &
```

Outputs are saved under:

```text
experiments/<run_name>/models/net_g_latest.pth
logs/diffmsr_<run_tag>.log
logs/generated_options/<run_tag>.yml
```

## 4. Test Models And Generate Figures

Run full held-out tests on the `valid/` split:

```bash
DATA_ROOT="$DATA_ROOT" TEST_GROUP=timesteps GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_pd_fs_full_tests.sh

DATA_ROOT="$DATA_ROOT" TEST_GROUP=latent GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_pd_fs_full_tests.sh
```

Useful `TEST_GROUP` values are `baseline`, `timesteps`, `latent`, `latent256`, and `all`.

If your timestep runs used a different shared Stage-1 checkpoint, pass it to the test wrapper:

```bash
DATA_ROOT="$DATA_ROOT" TEST_GROUP=timesteps TIMESTEP_STAGE1_CKPT=/path/to/stage1.pth \
  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_pd_fs_full_tests.sh
```

Generate metric plots, side-by-side examples, and compute-time plots:

```bash
"$PYTHON_BIN" scripts/visualize_pd_fs_ablation_results.py --top-n 5 --middle-slice-fraction 0.6
"$PYTHON_BIN" scripts/plot_pd_fs_compute_time.py
```

The main outputs are written to:

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

## Smoke Tests Before Commit

Run the lightweight smoke suite:

```bash
bash scripts/smoke_test_pd_fs_wrappers.sh
```

This checks shell syntax, `--help` paths, dry-run option generation for the shared denoising-ablation Stage 1, denoising Stage 2, latent-size Stage 1 + Stage 2, and full-test YAMLs. It validates that the generated YAMLs contain the requested dataset root, `prior_dim`, iteration count, denoising steps, and Stage-1 checkpoint path. It does not start real training.

If the raw DICOM database is available and you want to also test a tiny conversion:

```bash
RAW_DB=/path/to/knee_mri_clinical_seq_batch2 RUN_PREPROCESS_SMOKE=1 \
  bash scripts/smoke_test_pd_fs_wrappers.sh
```

Use an already extracted DICOM folder for this smoke test when possible. If `RAW_DB` is an archive, the smoke script skips extraction by default because unpacking the archive may expand the whole database; set `ALLOW_ARCHIVE_EXTRACT=1` only if that is intentional.

## Reported Results

All metrics below are on the held-out `valid/` split.

### Latent/prior size, 100k Stage 1 + 100k Stage 2, 8 denoising steps

| `prior_dim` | PSNR | SSIM | total wall time |
| ---: | ---: | ---: | ---: |
| 64 | 30.0977 | 0.8371 | 8.96 h |
| 128 | 30.1032 | 0.8372 | 9.20 h |
| 256 | 30.0834 | 0.8368 | 5.04 h |
| 512 | 30.0763 | 0.8367 | 9.51 h |

The 128-dimensional latent gave the best 100k latent-ablation score, but the differences between 64, 128, and 256 are small.

### Denoising steps, 100k Stage 2, shared 256-dimensional Stage 1

| denoising steps | PSNR | SSIM | Stage 2 + test wall time |
| ---: | ---: | ---: | ---: |
| 1 | 29.9231 | 0.8349 | 4.62 h |
| 2 | 30.1042 | 0.8372 | 4.74 h |
| 4 | 30.1055 | 0.8373 | 6.22 h |
| 6 | 30.1034 | 0.8372 | 4.43 h |
| 8 | 30.1064 | 0.8373 | 4.54 h |

The best timestep result was 8 denoising steps, with 4 steps essentially tied. One step was clearly worse. The 4-step runtime was an outlier in the logs; this training loop does not become linearly slower with the number of inference denoising steps.

### Full baseline

| model | PSNR | SSIM |
| --- | ---: | ---: |
| 256-dimensional latent, full Stage-2 run | 30.5910 | 0.8473 |

## Notes For Submission

- Raw DICOMs, converted `.mat` files, checkpoints, logs, and result images are intentionally ignored by git.
- The degradation used here follows the original DiffMSR demo style: center-cropped k-space and inverse FFT. It is a controlled super-resolution degradation, not a realistic Cartesian undersampling mask with aliasing.
- The validation split was not used for gradient updates. It was used for monitoring during training and then for final reporting because no separate test split was created.

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [SRFormer](https://github.com/HVision-NKU/SRFormer).
