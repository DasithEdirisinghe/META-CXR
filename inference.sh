#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate meta-cxr

python3 inference.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml