# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Criteo 100k
python3.6 scripts/preprocess.py \
              --sample-data-file /data/qq/nasrec_search/test_full_Criteo/criteo_processed.npz  \
              --save-filename /data/qq/nasrec_search/test_full_Criteo/criteo_100k  \
              --mode sample   \
              --num-samples 100000
