#!/usr/bin/env bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

DATA_DIR=dataset-open-world-vision-challenge/trainval
MODEL=moco2_sup_cotnet101_hybrid_se
python3 main_moco_sup.py \
  --data ${DATA_DIR} \
  -a cotnet101_hybrid_se \
  --model MoCoSup \
  --lr 0.1 \
  --batch-size 128 \
  --epochs 200 \
  --img-size 224 \
  --dist-url 'tcp://localhost:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --work-dirs workdirs/pretrained/${MODEL} \
  -p 50 \
  -s 10
