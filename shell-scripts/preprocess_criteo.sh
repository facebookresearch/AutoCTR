# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Criteo 1000
python3.6 scripts/preprocess.py \
              --dataset-name criteo \
              --data-file /data/qq/nasrec_search/test_full_Criteo \
              --mode raw
