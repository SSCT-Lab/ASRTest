#!/usr/bin/env bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#           2019   IIIT-Bangalore (Shreekantha Nadig)
# Apache 2.0.


create_glm_stm=false

if [ $# -le 0 ]; then
    echo "Argument should be the Timit directory, see ../run.sh for example."
    exit 1;
fi

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

if [ $2 ]; then
    if [[ $2 = "char" || $2 = "phn" ]]; then
        trans_type=$2
    else
        echo "Transcript type must be one of [phn, char]"
        echo $2
    fi
else
    trans_type=phn
fi

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=sph2pipe
if ! command -v "${sph2pipe}" &> /dev/null; then
    echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
    exit 1;
fi

[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased

# Now check what case the directory structure is
uppercased=true

# if [ -d $1/TRAIN-NEW ] || [ -d $1/TRAIN-ORGI ]; then
#     uppercased=true
# fi

# str1="$1"
# str2="orgi"

# result=$(echo $str1 | grep "${str2}")
# if [[ "$result" != "" ]] ; then
#     train=train-orgi
#     train_dir=TRAIN-ORGI
#     test_dir=TEST-ORGI
# else
#     train=train-new
#     train_dir=TRAIN-NEW
#     test_dir=TEST-NEW
# fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT
echo $tmpdir
# Get the list of speakers. The list of speakers in the 24-speaker core test
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
if $uppercased; then
    tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
#     ls -d "$1"/${train_dir}/DR*/* | sed -e "s:^.*/::" > $tmpdir/${train}_spk
else
    tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
#     ls -d "$1"/${train_dir}/dr*/* | sed -e "s:^.*/::" > $tmpdir/${train}_spk
fi

cd $dir
for x in dev ; do
    # First, find the list of audio files (use only si & sx utterances).
    # Note: train & test sets are under different directories, but doing find on
    # both and grepping for the speakers will work correctly.
    orgi_dir=${1/"TIMIT-new"/"TIMIT-orgi"}
    find $1/TEST-NOISE -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist
    find $1/TEST-FEATURE -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk >> ${x}_sph.flist
    find $1/TEST-ROOM -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk >> ${x}_sph.flist
    sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
    paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

    cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

    # Now, Convert the transcripts into our format (no normalization yet)
    # Get the transcripts: each line of the output contains an utterance
    # ID followed by the transcript.

 if [ $trans_type = "phn" ]
        then
            echo "phone transcript!"
            find $1/TEST-NOISE -not \( -iname 'SA*' \) -iname '*.PHN' \
            | grep -f $tmpdir/${x}_spk >> $tmpdir/${x}_phn.flist
            find $1/TEST-FEATURE -not \( -iname 'SA*' \) -iname '*.PHN' \
            | grep -f $tmpdir/${x}_spk >> $tmpdir/${x}_phn.flist
            find $1/TEST-ROOM -not \( -iname 'SA*' \) -iname '*.PHN' \
            | grep -f $tmpdir/${x}_spk >> $tmpdir/${x}_phn.flist
            sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i' $tmpdir/${x}_phn.flist \
            > $tmpdir/${x}_phn.uttids
            while read line; do
                [ -f $line ] || error_exit "Cannot find transcription file '$line'";
                cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
            done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
            paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
            | sort -k1,1 > ${x}.trans

        elif [ $trans_type = "char" ]
        then
            echo "char transcript!"
            find $1/TEST-NOISE -not \( -iname 'SA*' \) -iname '*.WRD' \
            | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_wrd.flist
            find $1/TEST-ROOM -not \( -iname 'SA*' \) -iname '*.WRD' \
            | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_wrd.flist
            find $1/TEST-FEATURE -not \( -iname 'SA*' \) -iname '*.WRD' \
            | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_wrd.flist
            sed -e 's:.*/\(.*\)/\(.*\).WRD$:\1_\2:i' $tmpdir/${x}_wrd.flist \
            > $tmpdir/${x}_wrd.uttids
            while read line; do
                [ -f $line ] || error_exit "Cannot find transcription file '$line'";
                cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z  A-Z]//g'
            done < $tmpdir/${x}_wrd.flist > $tmpdir/${x}_wrd.trans
            paste $tmpdir/${x}_wrd.uttids $tmpdir/${x}_wrd.trans \
            | sort -k1,1 > ${x}.trans
        else
            echo "WRONG!"
            echo $trans_type
            exit 0;
        fi

    # Do normalization steps.
    cat ${x}.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 39 | sort > $x.text || exit 1;

    # Create wav.scp
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp

    # Make the utt2spk and spk2utt files.
    cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk
    cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;

    # Prepare gender mapping
    cat $x.spk2utt | awk '{print $1}' | perl -ane 'chop; m:^.:; $g = lc($&); print "$_ $g\n";' > $x.spk2gender


    if "${create_glm_stm}"; then
        # Prepare STM file for sclite:
        wav-to-duration --read-entire-file=true scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
        awk -v dur=${x}_dur.ark \
        'BEGIN{
         while(getline < dur) { durH[$1]=$2; }
         print ";; LABEL \"O\" \"Overall\" \"Overall\"";
         print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
         print ";; LABEL \"M\" \"Male\" \"Male speakers\"";
       }
       { wav=$1; spk=wav; sub(/_.*/,"",spk); $1=""; ref=$0;
         gender=(substr(spk,0,1) == "f" ? "F" : "M");
         printf("%s 1 %s 0.0 %f <O,%s> %s\n", wav, spk, durH[wav], gender, ref);
       }
        ' ${x}.text >${x}.stm || exit 1

        # Create dummy GLM file for sclite:
        echo ';; empty.glm
      [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
        ' > ${x}.glm
    fi
done


echo "Data preparation succeeded"
