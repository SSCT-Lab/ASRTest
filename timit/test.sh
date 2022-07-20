#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

test_sets="test-feature"
model_name="kamo-naoyuki/timit_asr_train_asr_raw_word_valid.acc.ave"
asr_exp="exp/${model_name}"
selected_num=192
orig_flag=false
need_decode=true
stage=1
stop_stage=4
train_flag=false
dataset="TIMIT"

# Set this to one of ["phn", "char"] depending on your requirement
trans_type=phn
if [ "${trans_type}" = phn ]; then
    # If the transcription is "phn" type, the token splitting should be done in word level
    token_type=word
else
    token_type="${trans_type}"
fi

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm_rnn.yaml
inference_config=conf/decode_asr.yaml

. utils/parse_options.sh

./asr_test.sh \
    --token_type "${token_type}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --local_data_opts "--trans_type ${trans_type} --train_flag ${train_flag} --test_sets ${test_sets}" \
    --selected_num ${selected_num} \
    --orig_flag ${orig_flag} \
    --need_decode ${need_decode} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --asr_exp ${asr_exp} \
    --dataset ${dataset} 
