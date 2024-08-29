# [Micro-expression recognition using dual-branch 3DCNN network with novel attention mechanism]

## Chun-Ting Fang, Tsung-Jung Liu, Kuan-Hsien Liu  

***
> Abstract : Abstract—Micro-expressions refer to small and imperceptible changes in facial expressions displayed by humans in a very
short period of time, but these changes contain rich emotional information. In this paper, we propose a shallow dual-branch
3D-CNN backbone network architecture in the first stage as preliminary temporal and spatial feature learning. At the same
time, we have also enhanced and optimized the CAM (Channel Attention Module) within CBAM (Convolutional Block Attention
Module) so that it can be better applied to the extraction of subtle changes in micro-expression faces. Then we use GRU (gated
recurrent unit) and MSMH (multi-scale multi-head attention mechanism) as the second stage feature self-attention extraction.
The facial action unit (AU) is used to cut out key areas where micro-expressions occur to enhance the learning effect
of local features. In addition to testing on commonly used microexpression datasets, we also tested on lie detection datasets.
A large number of experimental results show that this method can achieve very good results with relatively simple input and
attention mechanisms.


## Network Architecture  

<table>
  <tr>
    <td colspan="2"><img src = "https://github.com/dannyFan-0201/Micro-Expression-Recognition-Using-A-Dual-Branch-3DCNN-Network/blob/main/img/model%20architecture.PNG" alt="CMFNet" width="1000"> </td>  
    
  </tr>
  
</table>

# Environment
- Python 3.9.0
- Tensorflow 2.10.1
- keras	2.10.0
- opencv-python	
- tensorboard	
- colorama
  
or see the requirements.txt

# How to try

## Download dataset (Most datasets require an application to download)
[SMIC] [SAMM] [CASME II] [CAS(ME)3] [Real-life deception detection Database]

## Set dataset path

Edit in Dual-Branch 3DCNN+AU (set path in config)

```python
output_folder ='./data/negative/training_frames' # This will be automatically generated.
negativepath = './data/negative/negative_video'
positivepath = './data/negative/positive_video'
surprisepath = './data/negative/surprise_video'

```

## Parameter settings

```python
excel_file_path = "/excel_file.xlsx"
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
df["Action Units"] = df["Action Units"].astype(str) #Convert Action Units data to string.
AU_CODE = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 43] #Select the Action Units you want.

```

## Run training
```python

python Dual-Branch 3DCNN+AU.py 

```
1. Set the dataset path.
2. Select the Action Units you want.
3. Set path and parameter details in model.
   
## Performance Evaluation

- MEGC2019 [SMIC Part] [SAMM Part] [CASME II Part]

<img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/Performance%20Evaluation.PNG"
  width="1312" height="250">

We compared our architecture with several other state of-the-art methods on the micro-expression datasets SMIC,SAMM and CASME II.
Both LOSO and MEGC2019 are used for performance comparison between our proposed method and SOTA methods.
(The best and second best scores are highlighted and underlined respectively.)
All training and testing base on same 4090.

- Evaluation experimental results based on the CAS(ME)3 dataset.
<img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/CAS(ME)3performance.PNG"
  width="600" height="150">

- Evaluation experimental results based on the lie detection dataset.
<img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/lie_detection.PNG"
  width="600" height="150">


## Ablation study

- SMIC DATASET ABLATION EXPERIMENT ON SINGLE-BRANCH 3DCNN INFRASTRUCTURE.
  <img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/ab1.PNG"
  width="600" height="150">

- SMIC DATASET ABLATION EXPERIMENT OF DUAL-BRANCH 3DCNN ARCHITECTURE.
  <img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/ab2.PNG"
  width="600" height="150">
  
- FOR THE ABLATION EXPERIMENT OF ADDING AU TO THE MODEL.
  <img src="https://github.com/dannyFan-0201/Micro-expression-recognition-using-dual-branch-3DCNN-network-with-novel-attention-mechanism/blob/main/img/ab3.PNG"
  width="600" height="150">


---
## Contact
If you have any question, feel free to contact danny80351@gmail.com

## Citation
```

