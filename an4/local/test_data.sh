#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100

datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/
ndev_utt=100
test_set=${1}
log "$0 $*"
echo ${test_set}
. utils/parse_options.sh

# if [ $# -ne 0 ]; then
#     log "Error: No positional arguments are required."
#     exit 2
# fi

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    mkdir -p ${datadir}
    local/download_and_untar.sh ${datadir} ${data_url}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/${test_set}

    if [[ ${test_set} =~ "130" ]]
    then 
        log "Do nut use preparing data"
    else
        python3 local/test_data_prep.py ${an4_root} ${test_set}
    fi
    for x in ${test_set}; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"

