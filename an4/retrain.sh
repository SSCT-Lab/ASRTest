#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
retrain_model="asr_train_raw_en_bpe30"
stage=1
stop_stage=10
train_set="retrain-gini"
valid_set="dev-gini"
. utils/parse_options.sh

./asr_retrain.sh \
    --lang en \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --use_lm false \
    --test_sets "test-new" \
    --retrain_model ${retrain_model} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --token_type bpe \
    --local_data_opts "--train_set ${train_set} --train_dev ${valid_set} " \
    --asr_args "--max_epoch 25 --batch_size 20 " \
    --lm_train_text "data/${train_set}/text" "$@" 
