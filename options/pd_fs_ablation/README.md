# PD/PD-FS Experiment YAMLs

These YAMLs are the source of truth for reproducing the reported PD/PD-FS ablation runs.

They assume:

```text
dataset root: mri_data_complex/mc_knee_pd_fs
checkpoints:  experiments/<run_name>/models/
```

## Baseline

```text
train/baseline_stage1_prior256_500k.yml
train/baseline_stage2_prior256_500k.yml
test/baseline_prior256_500k.yml
```

The logged baseline Stage 2 and test use:

```text
experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```

## Denoising-Step Ablation

Stage 2 training YAMLs:

```text
train/timesteps_steps1_stage2_100k.yml
train/timesteps_steps2_stage2_100k.yml
train/timesteps_steps4_stage2_100k.yml
train/timesteps_steps6_stage2_100k.yml
train/timesteps_steps8_stage2_100k.yml
```

Matching test YAMLs:

```text
test/timesteps_steps1_100k.yml
test/timesteps_steps2_100k.yml
test/timesteps_steps4_100k.yml
test/timesteps_steps6_100k.yml
test/timesteps_steps8_100k.yml
```

All timestep runs reuse the baseline Stage 1 checkpoint:

```text
experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```

## Latent-Prior-Size Ablation

```text
train/latent_prior64_stage1_100k.yml   -> train/latent_prior64_stage2_100k.yml   -> test/latent_prior64_100k.yml
train/latent_prior128_stage1_100k.yml  -> train/latent_prior128_stage2_100k.yml  -> test/latent_prior128_100k.yml
train/latent_prior256_stage1_100k.yml  -> train/latent_prior256_stage2_100k.yml  -> test/latent_prior256_100k.yml
train/latent_prior512_stage1_100k.yml  -> train/latent_prior512_stage2_100k.yml  -> test/latent_prior512_100k.yml
```

## Image-Space Partner Experiment

```text
train/train_S1_image.yml
train/train_S2_image.yml
test/test_image.yml
```

## Running One YAML

```bash
GPU_ID=3 PYTHON_BIN=/opt/miniconda3/bin/python \
  OPT=options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  bash train_S1.sh

GPU_ID=3 PYTHON_BIN=/opt/miniconda3/bin/python \
  OPT=options/pd_fs_ablation/test/latent_prior128_100k.yml \
  bash test.sh
```
