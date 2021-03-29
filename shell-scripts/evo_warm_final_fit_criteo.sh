# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


python3.6 scripts/final_fit.py \
        --data-set-name criteo --data-file ./example-data/criteo_500k.npz \
        --model-file ./results/evo/criteo/20200925-173209/M_74_S_73.json --save-model --save-model-path . \
        --batch-size 4096 --hash-size 10000 \
        --use-gpu --maxLoad 0.8 --maxMemory 0.8 --num-workers 20 --num-machines 1 --save-batches  \
        --total-gpus 8 --excludeID 0,1,2,3,4,5,6 --numpy-seed 123 --torch-seed 4321 --waiting-time 10

# 2>&1 | tee ./results/random/criteo/random.log
