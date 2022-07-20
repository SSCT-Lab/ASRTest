#!/usr/bin/env bash

# Copyright 2019 Nagoya University (Masao Someki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=3       # start from -1 if you need to start from data download
stop_stage=3
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
cmvn=
# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs
use_lang_model=false
lang_model=

# decoding parameter
p=0.005
recog_model=
recog_dir=
decode_config=
decode_dir=decode
api=v2

# bpemode (unigram or bpe)
nbpe=500
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

train_config=
decode_config=
preprocess_config=
lm_config=
models=tedlium3.conformer
# gini related
orig_flag=false
orig_dir=
need_decode=false

data_type=legacy
train_set=train_trim_sp
recog_set=dev-new

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# legacy setup


#adaptation setup
#data_type=speaker-adaptation
#train_set=train_adapt_trim_sp
#train_dev=dev_adapt_trim
# recog_set="dev_adapt test_adapt"

download_dir=${decode_dir}/download

if [ "${api}" = "v2" ] && [ "${backend}" = "chainer" ]; then
    echo "chainer backend does not support api v2." >&2
    exit 1;
fi

if [ -z $models ]; then
    if [ $use_lang_model = "true" ]; then
        if [[ -z $cmvn || -z $lang_model || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, lang_model, recog_model and decode_config are required.' >&2
            exit 1
        fi
    else
        if [[ -z $cmvn || -z $recog_model || -z $decode_config ]]; then
            echo 'Error: models or set of cmvn, recog_model and decode_config are required.' >&2
            exit 1
        fi
    fi
fi

dir=${download_dir}/${models}
mkdir -p ${dir}

# Download trained models
if [ -z "${cmvn}" ]; then
    #download_models
    cmvn=$(find ${download_dir}/${models} -name "cmvn.ark" | head -n 1)
fi
if [ -z "${lang_model}" ] && ${use_lang_model}; then
    #download_models
    lang_model=$(find ${download_dir}/${models} -name "rnnlm*.best*" | head -n 1)
fi
if [ -z "${recog_model}" ]; then
    #download_models
    if [ -z "${recog_dir}" ]; then
        recog_model=$(find ${download_dir}/${models} -name "model*.best*" | head -n 1)
    else
        recog_model=$(find "${recog_dir}/results" -name "model.loss.best" | head -n 1)
    fi
    echo "recog_model is ${recog_model}"
fi
if [ -z "${decode_config}" ]; then
    #download_models
    decode_config=$(find ${download_dir}/${models} -name "decode*.yaml" | head -n 1)
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${lang_model}" ] && ${use_lang_model}; then
    echo "No such language model: ${lang_model}"
    exit 1
fi
if [ ! -f "${recog_model}" ]; then
    echo "No such E2E model: ${recog_model}"
    exit 1
fi
if [ ! -f "${decode_config}" ]; then
    echo "No such config file: ${decode_config}"
    exit 1
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    if [ -z "${recog_dir}" ]; then 
        local/prepare_test_data.sh ${data_type} ${recog_set}
    fi
    for dset in ${recog_set}; do
        utils/fix_data_dir.sh data/${dset}.orig
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done
fi

# feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
# feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}/feats.scp ${cmvn} exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/train_trim_sp_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/train_trim_sp_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    # make json labels
     for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model\
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)

if [ -z "${recog_dir}" ]; then
    expname=${models}
    expdir=exp/${expname}
    mkdir -p ${expdir}
else
    expdir=${recog_dir}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    nj=32
    if ${use_lang_model}; then
        recog_opts="--rnnlm ${lang_model}"
    else
        recog_opts=""
    fi
    #feat_recog_dir=${decode_dir}/dump
    pids=() # initialize pids
    #trap 'rm -rf data/"${recog_set}" data/"${recog_set}.orig"' EXIT
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_decode_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

#         # split data
        if "${need_decode}"; then
            splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        fi
        orig_dir=${expdir}/decode_test-orig_decode_${lmtag}
        
        #### use CPU for decoding
        ngpu=0
        if "${need_decode}"; then
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_test.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split32utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${recog_model}  \
            --api ${api} \
            --orig_dir ${orig_dir} \
            --need_decode ${need_decode} \
            --orig_flag ${orig_flag} \
            --recog_set ${recog_set} \
            ${recog_opts}
            
        else
            asr_test.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${recog_model}  \
            --api ${api} \
            --orig_dir ${orig_dir} \
            --need_decode ${need_decode} \
            --orig_flag ${orig_flag} \
            --recog_set ${recog_set} \
            ${recog_opts}           
        fi

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Scoring"
    decode_dir=decode_${recog_set}_decode_${lmtag}
    if "${need_decode}"; then
       score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --need_decode ${need_decode} --guide_type "gini" ${expdir}/${decode_dir} ${dict}
    else
       score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true --need_decode ${need_decode} --guide_type "gini" ${expdir}/${decode_dir} ${dict}
    fi
    echo "Finished"
fi
