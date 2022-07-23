# AsrTest: Automated Testing for Deep-Neural-Network-Driven Speech Recognition Systems
We experiment ASRTest with the ASR toolkit ESPnet: (https://github.com/espnet/espnet). 

## generated data 
- We generated 8 times the size of the data, which can meet all kinds of model testing.
- AN4: https://pan.baidu.com/s/1KCDb1LKdmaExqqZqlrfAyw?pwd=xk95
- TIMIT: https://pan.baidu.com/s/12WJJLdnKejEVTW1Ft4DX2A?pwd=vg5w
- TEDLIUM2: https://pan.baidu.com/s/1lSo1zuf-QPwKz6MOeYN6iQ?pwd=pyqf
- TEDLIUM3: https://pan.baidu.com/s/1-21psFg2oZrt_RSgabdbHg?pwd=p8xu

## How to use the generated data for testing
- Install the ESPnet (https://github.com/espnet/espnet). 
  - You can install it easily by following the instruction provide by ESPnet.
- Add scripts to your espnet directory
  - For egs2 (Scripts are also available for other models in egs2): 
  - ASRTest/TEMPLATE/asr1/asr_test.sh ---> your espnet directory/egs2/TEMPLATE/asr1/asr_test.sh
  - ASRTest/an4/* ---> your espnet directory/egs2/an4/asr1/
  - ASRTest/timit/* ---> your espnet directory/egs2/timit/asr1/
  - For egs: 
  - ASRTest/tedlium2/* ---> your espnet directory/egs/tedlium2/asr1/
  - ASRTest/tedlium3/* ---> your espnet directory/egs/tedlium3/asr1/
