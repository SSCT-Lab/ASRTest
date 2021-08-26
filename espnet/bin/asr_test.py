#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model decoding script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool
from espnet.utils.gini_guide import *
from espnet.utils.gini_utils import *
from espnet.utils.retrain_utils import *

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`",
    )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="chainer",
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for beam search (0: means no batch processing)",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    parser.add_argument(
        "--api",
        default="v1",
        choices=["v1", "v2"],
        help="Beam search APIs "
        "v1: Default API. It only supports the ASRInterface.recognize method "
        "and DefaultRNNLM. "
        "v2: Experimental API. It supports any models that implements ScorerInterface.",
    )
    # task related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--result-label",
        type=str,
        required=True,
        help="Filename of result label data (json)",
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-spkrs",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of speakers in the speech",
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # search related
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--ctc-weight", type=float, default=0.0, help="CTC weight in joint decoding"
    )
    parser.add_argument(
        "--weights-ctc-dec",
        type=float,
        action="append",
        help="ctc weight assigned to each encoder during decoding."
        "[in multi-encoder mode only]",
    )
    parser.add_argument(
        "--ctc-window-margin",
        type=int,
        default=0,
        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""",
    )
    # transducer related
    parser.add_argument(
        "--search-type",
        type=str,
        default="default",
        choices=["default", "nsc", "tsd", "alsd"],
        help="""Type of beam search implementation to use during inference.
        Can be either: default beam search, n-step constrained beam search ("nsc"),
        time-synchronous decoding ("tsd") or alignment-length synchronous decoding
        ("alsd").
        Additional associated parameters: "nstep" + "prefix-alpha" (for nsc),
        "max-sym-exp" (for tsd) and "u-max" (for alsd)""",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=1,
        help="Number of expansion steps allowed in NSC beam search.",
    )
    parser.add_argument(
        "--prefix-alpha",
        type=int,
        default=2,
        help="Length prefix difference allowed in NSC beam search.",
    )
    parser.add_argument(
        "--max-sym-exp",
        type=int,
        default=2,
        help="Number of symbol expansions allowed in TSD decoding.",
    )
    parser.add_argument(
        "--u-max",
        type=int,
        default=400,
        help="Length prefix difference allowed in ALSD beam search.",
    )
    parser.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize transducer scores by length",
    )
    # rnnlm related
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument(
        "--word-rnnlm", type=str, default=None, help="Word RNNLM model file to read"
    )
    parser.add_argument(
        "--word-rnnlm-conf",
        type=str,
        default=None,
        help="Word RNNLM model config file to read",
    )
    parser.add_argument("--word-dict", type=str, default=None, help="Word list to read")
    parser.add_argument("--lm-weight", type=float, default=0.1, help="RNNLM weight")
    # ngram related
    parser.add_argument(
        "--ngram-model", type=str, default=None, help="ngram model file to read"
    )
    parser.add_argument("--ngram-weight", type=float, default=0.1, help="ngram weight")
    parser.add_argument(
        "--ngram-scorer",
        type=str,
        default="part",
        choices=("full", "part"),
        help="""if the ngram is set as a part scorer, similar with CTC scorer,
                ngram scorer only scores topK hypethesis.
                if the ngram is set as full scorer, ngram scorer scores all hypthesis
                the decoding speed of part scorer is musch faster than full one""",
    )
    # streaming related
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default=None,
        choices=["window", "segment"],
        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""",
    )
    parser.add_argument("--streaming-window", type=int, default=10, help="Window size")
    parser.add_argument(
        "--streaming-min-blank-dur",
        type=int,
        default=10,
        help="Minimum blank duration threshold",
    )
    parser.add_argument(
        "--streaming-onset-margin", type=int, default=1, help="Onset margin"
    )
    parser.add_argument(
        "--streaming-offset-margin", type=int, default=1, help="Offset margin"
    )
    # non-autoregressive related
    # Mask CTC related. See https://arxiv.org/abs/2005.08700 for the detail.
    parser.add_argument(
        "--maskctc-n-iterations",
        type=int,
        default=10,
        help="Number of decoding iterations."
        "For Mask CTC, set 0 to predict 1 mask/iter.",
    )
    parser.add_argument(
        "--maskctc-probability-threshold",
        type=float,
        default=0.999,
        help="Threshold probability for CTC output",
    )
    parser.add_argument("--orgi_flag", type=str, default="False")
    parser.add_argument("--orgi_dir", type=str, required=True)
    parser.add_argument(
        "--recog_set",
        type=str,
        help="Whether use gini or not",
    )
    parser.add_argument(
        "--need_decode",
        type=str,
        default="True",
    )
    
    return parser


def str_2_bool(s):
    return True if s.lower() =='true' else False

def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)
    orgi_flag = str_2_bool(args.orgi_flag)
    need_decode = str_2_bool(args.need_decode)
    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error(
            "It seems that both --rnnlm and --word-rnnlm are specified. "
            "Please use either option."
        )
        sys.exit(1)

    # recog
    logging.info("backend = " + args.backend)
    if args.num_spkrs == 1:
        if args.backend == "chainer":
            from espnet.asr.chainer_backend.asr import recog
            recog(args)
        elif args.backend == "pytorch":
            if args.num_encs == 1:
                # Experimental API that supports custom LMs
                if args.api == "v2":
                    if need_decode:
                        from espnet.asr.pytorch_backend.recog_test import recog_v2
                        recog_v2(args, orgi_flag)
                    else:
                        new_dir = "/".join(args.result_label.split("/")[0:-1])
                        orgi_dir = new_dir.replace("new", "orig")
                        if "test" in args.recog_set:
                            aug_type = args.recog_set.split("-")[-1]
                            gini_audio = sort_test_by_avg_gini_v1(args.orgi_dir, new_dir, aug_type)
                            cov_audio = sort_test_by_cov_v1(args.orgi_dir, new_dir, aug_type)
                            random_audio = random_samples_v1(args.orgi_dir, new_dir)
                            test_data_prep("data/test-orgi", "data/" + args.recog_set, "ted2", gini_audio, "gini", 1155, args.recog_set)
                            test_data_prep("data/test-orgi", "data/" + args.recog_set, "ted2", random_audio, "random", 1155, args.recog_set)
                            test_data_prep("data/test-orgi", "data/" + args.recog_set, "ted2", cov_audio, "cov", 1155, args.recog_set)
                        if "train-new" in args.recog_set:
                            mix_gini, mix_cov = mix_train_set(orgi_dir, new_dir, "ted3", 4685, 0.01)
                            random_audio = random_retrain(new_dir, mix_gini, 0.25)
                            gini_audio = avg_gini_retrain(new_dir, mix_gini, 0.25)
                            cov_audio = cov_retrain(new_dir, mix_cov, 0.25)
                            orgi_data = "data/train-orig"
                            new_data = "data/" + args.recog_set
                            retrain_data_prep(orgi_data, new_data, "ted2", gini_audio, "gini")
                            retrain_data_prep(orgi_data, new_data, "ted2", random_audio, "random")
                            retrain_data_prep(orgi_data, new_data, "ted2", cov_audio, "cov")
                        if "dev-new" in args.recog_set:
                            mix_gini, mix_cov = mix_train_set(orgi_dir, new_dir, "ted2", 254, 0.01)
                            random_audio = random_retrain(new_dir, mix_gini, 0.25)
                            gini_audio = avg_gini_retrain(new_dir, mix_gini, 0.25)
                            cov_audio = cov_retrain(new_dir, mix_cov, 0.25)
                            orgi_data = "data/dev-orig"
                            new_data = "data/dev-new"
                            retrain_data_prep(orgi_data, new_data, "ted2", gini_audio, "gini")
                            retrain_data_prep(orgi_data, new_data, "ted2", random_audio, "random")
                            retrain_data_prep(orgi_data, new_data, "ted2", cov_audio, "cov")
                else:
                    from espnet.asr.pytorch_backend.asr import recog

                    if args.dtype != "float32":
                        raise NotImplementedError(
                            f"`--dtype {args.dtype}` is only available with `--api v2`"
                        )
                    recog(args)
            else:
                if args.api == "v2":
                    raise NotImplementedError(
                        f"--num-encs {args.num_encs} > 1 is not supported in --api v2"
                    )
                else:
                    from espnet.asr.pytorch_backend.asr import recog

                    recog(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    elif args.num_spkrs == 2:
        if args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_mix import recog

            recog(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
