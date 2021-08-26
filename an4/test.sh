#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

test_sets="test-noise"
test_model="retrain-gini"
orgi_flag=false
need_decode=true
selected_num=130
stage=1
stop_stage=4
dataset="an4"
. utils/parse_options.sh

./asr_test.sh \
    --use_lm false \
    --test_sets ${test_sets} \
    --local_data_opts ${test_sets}\
    --orgi_flag ${orgi_flag} \
    --need_decode ${need_decode} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --token_type bpe \
    --asr_exp "exp/asr_train_raw_en_bpe30" \
    --dataset ${dataset} \
    --selected_num ${selected_num}
