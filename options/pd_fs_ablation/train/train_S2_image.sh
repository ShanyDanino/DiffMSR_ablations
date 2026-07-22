#!/bin/bash
export PYTHONPATH="$(pwd)"
CUDA_VISIBLE_DEVICES=0 python DiffMSR_Main/train.py -opt options/image/train_DiffMSR_S2_x4_image.yml
