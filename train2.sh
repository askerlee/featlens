#!/bin/bash
python3.7 -m torch.distributed.launch --nproc_per_node=2 train.py --bs 512 "$@"
