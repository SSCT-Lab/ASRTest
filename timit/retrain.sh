#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

retrain_model="kamo-naoyuki/timit_asr_train_asr_raw_word_valid.acc.ave"
stage=1
stop_stage=10
train_set="retrain-cov"
train_dev="dev-cov"
test_sets="test-noise test-feature test-room"

# Set this to one of ["phn", "char"] depending on your requirement
trans_type=phn
if [ "${trans_type}" = phn ]; then
    # If the transcription is "phn" type, the token splitting should be done in word level
    token_type=word
else
    token_type="${trans_type}"
fi

asr_config=conf/tuning/train_asr_rnn.yaml
lm_config=conf/train_lm_rnn.yaml
inference_config=conf/decode_asr.yaml
. utils/parse_options.sh

./asr_retrain.sh \
    --token_type "${token_type}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --local_data_opts "--trans_type ${trans_type} --train_set ${train_set} " \
    --lm_train_text "data/${train_set}/text" "$@" \
    --retrain_model "${retrain_model}" \
    --stage ${stage} \
    --asr_config "${asr_config}" \
    --stop_stage ${stop_stage} \
    --asr_args "--max_epoch 40 " 
#     --lm_config "${lm_config}" \
#     --inference_config "${inference_config}" \

