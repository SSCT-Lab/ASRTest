#!/usr/bin/env bash
# Copyright IIIT-Bangalore (Shreekantha Nadig)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
train_flag=false
trans_type=phn
test_sets=
log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh
. ./db.sh

echo "train flag is ${train_flag}"

# general configuration
if [ -z "${TIMIT}" ]; then
    log "Fill the value of 'TIMIT' of db.sh"
    exit 1
fi

log "data preparation started"
#TIMIT=/root/espnet/egs2/timit/asr1/data/local/data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    log "stage1: Preparing data for TIMIT for ${trans_type} level transcripts"
    echo $TIMIT
    if [[ ${test_sets} =~ "192" ]]
    then 
        log "Do nut use preparing data"
    else
        local/timit_test_data_prep.sh ${TIMIT} ${trans_type} ${train_flag} ${test_sets}
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage2: Formatting TIMIT directories"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    local/timit_format_test_data.sh ${train_flag} ${test_sets}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
