# ASRTest: Automated Testing for Deep-Neural-Network-Driven Speech Recognition Systems
We experiment ASRTest with the ASR toolkit ESPnet: (https://github.com/espnet/espnet). 

## generated data 
- We generated 8 times the size of the data, which can meet all kinds of model testing.
- AN4: https://pan.baidu.com/s/1KCDb1LKdmaExqqZqlrfAyw?pwd=xk95
- TIMIT: https://pan.baidu.com/s/12WJJLdnKejEVTW1Ft4DX2A?pwd=vg5w
- TEDLIUM2: https://pan.baidu.com/s/1lSo1zuf-QPwKz6MOeYN6iQ?pwd=pyqf
- TEDLIUM3: https://pan.baidu.com/s/1-21psFg2oZrt_RSgabdbHg?pwd=p8xu

## How to use the generated data for testing
### Install the ESPnet (https://github.com/espnet/espnet). 
- You can install it easily by following the instruction provide by ESPnet.
### Add scripts to your espnet directory
- For guidance utils:
  - ASRTest/utils/* ---> your espnet directory/utils
  - ASRTest/espnet/* ---> your espnet directory/espnet
  - ASRTest/espnet2/* ---> your espnet directory/espnet2
- For egs2 (Scripts are also available for other models in egs2): 
  - ASRTest/TEMPLATE/asr1/asr_test.sh ---> your espnet directory/egs2/TEMPLATE/asr1/asr_test.sh
  - ASRTest/an4/* ---> your espnet directory/egs2/an4/asr1/
  - ASRTest/timit/* ---> your espnet directory/egs2/timit/asr1/
- For egs: 
  - ASRTest/tedlium2/* ---> your espnet directory/egs/tedlium2/asr1/
  - ASRTest/tedlium3/* ---> your espnet directory/egs/tedlium3/asr1/
### Decode the generated data
- For egs2：
  - cd egs2/xxx/asr1/
  - ln -s ../../TEMPLATE/asr1/asr_test.sh ./asr_test.sh 
  - decode the origial test set: sh test.sh --test_sets "test-orig" --need_decode true --orig_flag true --dataset "xxx" --stage 1 --stop_stage 4 
  - decode the all transformed test set: sh test.sh --test_sets "xxxx" --need_decode true --orig_flag false --dataset "xxx" --stage 1 --stop_stage 4 
  - obtain the result on the test set transformed by ASRTest: sh test.sh --test_sets "xxxx" --need_decode false --orig_flag false --dataset "xxx" --stage 3 --stop_stage 4 
- For egs：
  - cd egs/xxx/asr1/
  - decode the origial test set: sh test.sh --recog_set "test-orig" --need_decode true --orig_flag true --stage 0 --stop_stage 4 
  - decode the transformed test set: sh test.sh --recog_set "xxxx" --need_decode true --orig_flag false --stage 0 --stop_stage 4 
  - obtain the result on the test set transformed by ASRTest: sh test.sh --recog_set "xxxx" --need_decode false --orig_flag false --stage 3 --stop_stage 4 
- test_set/recog_set: test-feature, test-noise, test-room, test-orig
- orig_flag: if test_set/recog_set is test-orig, orig_flag is true; otherwise, it is false
