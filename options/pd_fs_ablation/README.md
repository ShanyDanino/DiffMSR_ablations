# PD/PD-FS Ablation YAMLs

These are tracked copies of the experiment option files used for the reported PD/PD-FS ablations.

They assume the prepared dataset is at:

```text
mri_data_complex/mc_knee_pd_fs
```

and that checkpoints are saved under the `experiments/` paths referenced inside each YAML.

## Training YAMLs

Baseline:

- `train/baseline_stage1_prior256_500k.yml`
- `train/baseline_stage2_prior256_500k.yml`

Denoising-step ablation:

- `train/timesteps_steps1_stage2_100k.yml`
- `train/timesteps_steps2_stage2_100k.yml`
- `train/timesteps_steps4_stage2_100k.yml`
- `train/timesteps_steps6_stage2_100k.yml`
- `train/timesteps_steps8_stage2_100k.yml`

These Stage-2 timestep runs reuse the shared Stage-1 checkpoint:

```text
experiments/pd_fs_stage1_x4/models/net_g_330000.pth
```

Latent/prior-size ablation:

- `train/latent_prior64_stage1_100k.yml`
- `train/latent_prior64_stage2_100k.yml`
- `train/latent_prior128_stage1_100k.yml`
- `train/latent_prior128_stage2_100k.yml`
- `train/latent_prior256_stage1_100k.yml`
- `train/latent_prior256_stage2_100k.yml`
- `train/latent_prior512_stage1_100k.yml`
- `train/latent_prior512_stage2_100k.yml`

## Test YAMLs

- `test/baseline_prior256_500k.yml`
- `test/timesteps_steps1_100k.yml`
- `test/timesteps_steps2_100k.yml`
- `test/timesteps_steps4_100k.yml`
- `test/timesteps_steps6_100k.yml`
- `test/timesteps_steps8_100k.yml`
- `test/latent_prior64_100k.yml`
- `test/latent_prior128_100k.yml`
- `test/latent_prior256_100k.yml`
- `test/latent_prior512_100k.yml`

The test YAMLs still contain an inherited `train:` block from the original DiffMSR template. It is not used by `DiffMSR_Main/test.py`; the important evaluation fields are `datasets.test_1`, `network_g`, `network_S1`, and `path`.

## Run Directly

Train with one exact YAML:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/train.py \
  -opt options/pd_fs_ablation/train/latent_prior128_stage1_100k.yml \
  --launcher none
```

Test with one exact YAML:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u DiffMSR_Main/test.py \
  -opt options/pd_fs_ablation/test/latent_prior128_100k.yml \
  --launcher none
```

The wrapper scripts in `scripts/` are still the recommended way to run new variants, because they can rewrite `DATA_ROOT`, `TOTAL_ITER`, `TIMESTEPS`, and `PRIOR_DIM` without hand-editing YAML.
