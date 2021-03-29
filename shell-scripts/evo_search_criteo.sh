# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3.6 scripts/search.py \
        --data-set-name criteo --data-file ./example-data/criteo_100k.npz \
        --macro-space-type 2 --micro-space-type micro_mlp \
        --fbl-kill-time 100 --max-num-block 7 --batch-size 4096 --hash-size 10000 \
        --use-gpu --maxLoad 0.8 --maxMemory 0.8 --num-workers 20 --num-machines 6 --save-batches  \
        --searcher-type evo  --search-nepochs 100 --population-size 20 --survival-type comb --candidate-size 10 --sampler-type 100 \
        --total-gpus 8 --excludeID 0,1,2,3,4,5,6 --numpy-seed 123 --torch-seed 4321 --waiting-time 10 \
