#!/usr/bin/env python3

import os
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python test_data_prep.py [an4_root] [test_dir]")
    sys.exit(1)
an4_root = sys.argv[1]
test_set = sys.argv[2]
wav_dir = {"test": test_set}
for x in ["test"]:
    with open(
        os.path.join(an4_root, "etc", "an4_" + test_set + ".transcription")
    ) as transcript_f, open(os.path.join("data", wav_dir[x], "text"), "w") as text_f, open(
        os.path.join("data", wav_dir[x], "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", wav_dir[x], "utt2spk"), "w"
    ) as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        lines = sorted(transcript_f.readlines(), key=lambda s: s.split(" ")[0])
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = re.search(r"^(.*) \(", line).group(1)
            if words[:4] == "<s> ":
                words = words[4:]
            if words[-5:] == " </s>":
                words = words[:-5]
            source = re.search(r"\((.*)\)", line).group(1)
            pre, mid, last = source.split("-")
            utt_id = "-".join([mid, pre, last])
            text_f.write(utt_id + " " + words + "\n")
            if "test-orgi" in test_set:
                wav_scp_f.write(
                    utt_id
                    + " "
                    + "sph2pipe"
                    + " -f wav -p -c 1 "
                    + os.path.join(an4_root, "wav", wav_dir[x], mid, source + ".sph")
                    + " |\n"
                )
            elif ("dev-orig" in test_set) | ("train-orig" in test_set):
                wav_scp_f.write(
                    utt_id
                    + " "
                    + os.path.join(an4_root, "wav", wav_dir[x], mid, source + ".wav")
                    + "\n"
                )
            else:
                wav_scp_f.write(
                    utt_id
                    + " "
                    + os.path.join(an4_root, "wav", wav_dir[x], mid, source)
                    + "\n"
                )
            utt2spk_f.write(utt_id + " " + mid + "\n")

