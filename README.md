# DiffMSR PD/PD-FS Reproduction

This repository adapts DiffMSR/McDiff for knee MRI super-resolution using proton-density images as the target and proton-density fat-suppressed images as the reference.

Original paper: Guangyuan Li et al., "Rethinking Diffusion Model for Multi-Contrast MRI Super-Resolution", CVPR 2024.

## Data Mapping

The DiffMSR dataset code expects variables named `T1` and `T2`. For this project:

| DiffMSR field | Our data |
| --- | --- |
| `T2` | high-resolution PD target |
| `T2_128`, `T2_64` | k-space downsampled PD input |
| `T1`, `T1_128`, `T1_64` | PD fat-suppressed reference |

Raw data, converted `.mat` files, checkpoints, logs, and result images are not tracked by git.

## Environment

From the repository root:

```bash
bash scripts/create_diffmsr_env.sh
export PYTHON_BIN="$PWD/.conda/envs/diffmsr/bin/python"
export GPU_ID=3
export PYTHONPATH="$PWD"
"$PYTHON_BIN" scripts/check_diffmsr_env.py
```

If you already have the working cluster environment, this also works:

```bash
export PYTHON_BIN=/opt/miniconda3/bin/python
export GPU_ID=3
export PYTHONPATH="$PWD"
```

## 1. Prepare the Dataset

If the DICOM database is an archive, extract it first:

```bash
mkdir -p data/extracted_pd_fs
tar -xf /path/to/knee_DICOMs_batch2.tar.xz -C data/extracted_pd_fs
```

Convert the extracted DICOM tree into DiffMSR `.mat` files:

```bash
"$PYTHON_BIN" scripts/preprocess_pd_fs_dicom.py \
  --input-root data/extracted_pd_fs/knee_DICOMs_batch2 \
  --out-root mri_data_complex/mc_knee_pd_fs \
  --train-ratio 0.8 \
  --seed 0 \
  --overwrite
```

For our run, the prepared dataset statistics were:

| Property | Value |
| --- | ---: |
| paired studies | 1857 |
| train studies | 1486 |
| held-out valid/test studies | 371 |
| train slices | 40372 |
| held-out valid/test slices | 10119 |
| skipped slices | 54 |
| HR / mid / LR sizes | 256 / 128 / 64 |

The `valid/` split was not used for fitting weights. We used it as the held-out test set for the reported numbers.

Degradation: each slice is normalized, transformed to centered k-space, center-cropped to `256`, `128`, and `64`, and transformed back with inverse FFT. The script also copies the original DiffMSR demo mask to `mri_data_complex/mc_knee_pd_fs/dc_mask.mat`.

## 2. How to Run a YAML

The simplest launchers are:

```bash
bash train_S1.sh
bash train_S2.sh
bash test.sh
```

Each launcher runs one YAML. Override it with `OPT=...`:

```bash
GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  OPT=options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  bash train_S1.sh
```

Run in the background with `nohup`:

```bash
mkdir -p logs
nohup env GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  OPT=options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  bash train_S1.sh > logs/latent128_s1.nohup.log 2>&1 &
tail -f logs/latent128_s1.nohup.log
```

## 3. Reproduce the Reported Experiments

All exact YAMLs are under:

```text
options/pd_fs_ablation/train/
options/pd_fs_ablation/test/
```

### Experiment A: Full Baseline

This is the full 256-dimensional latent prior baseline.

```bash
nohup env GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  OPT=options/pd_fs_ablation/train/baseline_stage1_prior256_500k.yml \
  bash train_S1.sh > logs/baseline_s1.nohup.log 2>&1 &
```

Stage 2 for the logged baseline uses the Stage 1 checkpoint `experiments/pd_fs_stage1_x4/models/net_g_330000.pth`, because that is the checkpoint used in the completed run:

```bash
nohup env GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  OPT=options/pd_fs_ablation/train/baseline_stage2_prior256_500k.yml \
  bash train_S2.sh > logs/baseline_s2.nohup.log 2>&1 &
```

Then test:

```bash
nohup env GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  OPT=options/pd_fs_ablation/test/baseline_prior256_500k.yml \
  bash test.sh > logs/baseline_test.nohup.log 2>&1 &
```

### Experiment B: Denoising Steps

This ablation retrains only Stage 2. All runs reuse the same Stage 1 checkpoint:

```text
experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```

Sequential command:

```bash
for steps in 1 2 4 6 8; do
  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
    OPT="options/pd_fs_ablation/train/timesteps_steps${steps}_stage2_100k.yml" \
    bash train_S2.sh

  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
    OPT="options/pd_fs_ablation/test/timesteps_steps${steps}_100k.yml" \
    bash test.sh
done
```

Background sequential command:

```bash
nohup bash -lc '
set -euo pipefail
export GPU_ID=3
export PYTHON_BIN=/opt/miniconda3/bin/python
for steps in 1 2 4 6 8; do
  OPT="options/pd_fs_ablation/train/timesteps_steps${steps}_stage2_100k.yml" bash train_S2.sh
  OPT="options/pd_fs_ablation/test/timesteps_steps${steps}_100k.yml" bash test.sh
done
' > logs/timesteps_100k_sequence.nohup.log 2>&1 &
```

### Experiment C: Latent Prior Size

This ablation retrains both Stage 1 and Stage 2 for each latent/prior size.

```bash
for dim in 64 128 256 512; do
  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
    OPT="options/pd_fs_ablation/train/latent_prior${dim}_stage1_100k.yml" \
    bash train_S1.sh

  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
    OPT="options/pd_fs_ablation/train/latent_prior${dim}_stage2_100k.yml" \
    bash train_S2.sh

  GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
    OPT="options/pd_fs_ablation/test/latent_prior${dim}_100k.yml" \
    bash test.sh
done
```

The 256-dimensional 100k run can also be reproduced with the pipeline script we used:

```bash
nohup env GPU_ID="$GPU_ID" PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT=mri_data_complex/mc_knee_pd_fs \
  bash scripts/run_prior256_100k_pipeline.sh \
  > logs/prior256_100k_pipeline.nohup.log 2>&1 &
```

### Experiment D: Latent-Space vs Image-Space Diffusion

This section is for the partner experiment comparing the original latent-space prior against an image-space diffusion variant.

Existing image-space YAMLs:

```text
options/pd_fs_ablation/train/train_S1_image.yml
options/pd_fs_ablation/train/train_S2_image.yml
options/pd_fs_ablation/test/test_image.yml
```

Fill in the final image-space result here when it is finalized:

| Variant | Training YAML | Test YAML | PSNR | SSIM |
| --- | --- | --- | ---: | ---: |
| latent-space diffusion | baseline Stage 1 + baseline Stage 2 | `baseline_prior256_500k.yml` | 30.5910 | 0.8473 |
| image-space diffusion | TODO | TODO | TODO | TODO |

## 4. Generate Figures

After running tests, result files are expected under:

```text
results/<run_name>/visualization/*.mat
```

Generate PSNR/SSIM plots and side-by-side example images:

```bash
"$PYTHON_BIN" scripts/visualize_pd_fs_ablation_results.py \
  --top-n 5 \
  --middle-slice-fraction 0.6
```

Generate compute-time plots from the logs:

```bash
"$PYTHON_BIN" scripts/plot_pd_fs_compute_time.py
```

Main outputs:

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

For the latent-vs-image-space comparison, make sure the image-space test writes a result folder such as `results/pd_fs_test_image_space/visualization/*.mat`, then run:

```bash
"$PYTHON_BIN" scripts/visualize_pd_fs_ablation_results.py \
  --experiments space \
  --space-models latent=pd_fs_test_baseline_prior256_500k,image=pd_fs_test_image_space \
  --top-n 5 \
  --middle-slice-fraction 0.6
```

## 5. Reported Results

All metrics are on the held-out `valid/` split.

### Denoising Steps, 100k Stage 2

| steps | PSNR | SSIM | Stage 2 + test time |
| ---: | ---: | ---: | ---: |
| 1 | 29.9231 | 0.8349 | 4.62 h |
| 2 | 30.1042 | 0.8372 | 4.74 h |
| 4 | 30.1055 | 0.8373 | 6.22 h |
| 6 | 30.1034 | 0.8372 | 4.43 h |
| 8 | 30.1064 | 0.8373 | 4.54 h |

Best PSNR was 8 steps. Four steps was essentially tied and is the original DiffMSR setting.

### Latent Prior Size, 100k Stage 1 + 100k Stage 2

| `prior_dim` | PSNR | SSIM | total time |
| ---: | ---: | ---: | ---: |
| 64 | 30.0977 | 0.8371 | 8.96 h |
| 128 | 30.1032 | 0.8372 | 9.20 h |
| 256 | 30.0834 | 0.8368 | 5.04 h |
| 512 | 30.0763 | 0.8367 | 9.51 h |

The best 100k latent-size result was `prior_dim=128`, but the differences among 64/128/256 were small.

### Full Baseline

| model | PSNR | SSIM |
| --- | ---: | ---: |
| 256-dimensional latent prior, full Stage 2 | 30.5910 | 0.8473 |

## 6. Quick Checks Before Commit

These checks do not start training:

```bash
bash -n train_S1.sh train_S2.sh test.sh
bash -n scripts/run_diffmsr_nohup.sh scripts/run_pd_fs_full_tests.sh scripts/run_prior256_100k_pipeline.sh
"$PYTHON_BIN" -m py_compile \
  scripts/preprocess_pd_fs_dicom.py \
  scripts/visualize_pd_fs_ablation_results.py \
  scripts/plot_pd_fs_compute_time.py
```

Tiny preprocessing check:

```bash
"$PYTHON_BIN" scripts/preprocess_pd_fs_dicom.py \
  --input-root data/extracted_pd_fs/knee_DICOMs_batch2 \
  --out-root /tmp/diffmsr_pd_fs_tiny \
  --max-studies 3 \
  --max-samples 12 \
  --overwrite
```

## Notes

- The checked YAML files assume `mri_data_complex/mc_knee_pd_fs` as the dataset root.
- If you use a different Stage 1 checkpoint, update the `pretrain_network_S1` and `pretrain_network_g` paths in the relevant YAML.
- Test YAMLs contain an inherited `train:` block from the original template. `DiffMSR_Main/test.py` does not train from that block; it uses the dataset, network, and checkpoint fields.

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [DiffIR](https://github.com/Zj-BinXia/DiffIR), and [SRFormer](https://github.com/HVision-NKU/SRFormer).
