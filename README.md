# LLaVA-application-research
LLaVA rebuild and application of cross-modality generation

### 1. Set-up
Follow LLaVA original instructions: https://github.com/haotian-liu/LLaVA/tree/main

1. Clone this repository
```bash
git clone https://github.com/XinruiXiong/LLaVA-application-research.git
cd LLaVA-application-research
```
2. Install Package
```Shell
conda create -n llava-eeg python=3.10 -y
conda activate llava-eeg
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
Or download the wheel manually on github: https://github.com/Dao-AILab/flash-attention/releases if building failed (highly likely to happen).

### 2. Dataset
Download the [dataset](https://drive.google.com/file/d/1uW1opK8Hus2FpEyLMlND3SeLRjBbn5bG/view?usp=sharing), and unzip it to ./playground/data.


### 3. Finetune in the subset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/finetune_lora.sh 
```
The Lora version requires less GPU memory (4 A6000 GPU will be ok). For better hardware equipments, try ./scripts/v1_5/finetune.sh. 


### 4. Finetune in the EEG dataset
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/finetune_lora_eeg.sh 
```

### 5. Checkpoints download
[a link, I'll upload soon]

On Huggingface:

https://huggingface.co/Xinrui01/llava-13b-lora 

https://huggingface.co/Xinrui01/llava-13b-eeg-lora

### 6. Inference and Evaluate of the EEG-Text fine-tuning
Inference:
```bash
CUDA_VISIBLE_DEVICES=0 python inference-eeg.py
```
One GPU task, You may change the path in this file to select a specific eeg image.

Evaluate:
download [Test data Index](https://drive.google.com/file/d/1vjAX34KEA8mLq-5Yqy3R96HLff-5Un3b/view?usp=drive_link) and put under ./playground/data/

```bash
CUDA_VISIBLE_DEVICES=0 python eeg_predict.py
```
One GPU task, you'll get the output prediction in a csv file.

Then
```bash
python eval.py
```


You may find the model can't have a texual output (recieve only a "\</s\>") for some eeg images.In this case, due to the simple prompt, the model considers the best answer is no answer.

Theoritically, a better Prompt may solve this (we simply use "Please describe the EEG image." for every eeg image). But the fundamental reason is because the poor Visual Instruct Tuning file (one round conversation, limited dataset size, non-precise precise image precise description) failed to help model learn the pattern of the dataset.
We'll improve this part in the future work.




