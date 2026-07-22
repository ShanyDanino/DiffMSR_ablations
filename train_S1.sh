#!/bin/bash

# Default values
DOMAIN_MODE="latent_space"
PRIOR_DIM="256"
TIMESTEPS="4"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --domain_mode) DOMAIN_MODE="$2"; shift ;;
        --prior_dim) PRIOR_DIM="$2"; shift ;;
        --timesteps) TIMESTEPS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$DOMAIN_MODE" == "image_space" ]]; then
    YAML_FILE="train_S1_image.yml"
elif [[ "$DOMAIN_MODE" == "latent_space" ]]; then
    if [[ "$PRIOR_DIM" != "256" ]]; then
        case "$PRIOR_DIM" in
            64) YAML_FILE="latent_prior64_stage1_100k.yml" ;;
            128) YAML_FILE="latent_prior128_stage1_100k.yml" ;;
            512) YAML_FILE="latent_prior512_stage1_100k.yml" ;;
            *) echo "Error: No matching YAML file for prior_dim='${PRIOR_DIM}'"; exit 1 ;;
        esac
    elif [[ "$PRIOR_DIM" == "256" ]]; then
        case "$TIMESTEPS" in
            1) YAML_FILE="timesteps_steps1_stage1_100k.yml" ;;
            2) YAML_FILE="timesteps_steps2_stage1_100k.yml" ;;
            4) YAML_FILE="timesteps_steps4_stage1_100k.yml" ;;
            6) YAML_FILE="timesteps_steps6_stage1_100k.yml" ;;
            8) YAML_FILE="timesteps_steps8_stage1_100k.yml" ;;
            *) echo "Error: No matching YAML file for timesteps='${TIMESTEPS}'"; exit 1 ;;
        esac
    else
        # Default latent case (prior_dim=256, timesteps=4)
        YAML_FILE="latent_prior256_stage1_100k.yml"
    fi
else
    echo "Error: Unknown domain_mode='${DOMAIN_MODE}'"
    exit 1
fi

OPT_PATH="options/pd_fs_ablation/train/${YAML_FILE}"

if [[ ! -f "$OPT_PATH" ]]; then
    echo "Error: Configuration file not found at $OPT_PATH"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python DiffMSR_Main/train.py -opt "$OPT_PATH"
