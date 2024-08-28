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

## Qualitative comparisons

- Places2

<img src="https://i.imgur.com/FMGm4mB.jpg" width="1000" style="zoom:100%;">

Qualitative results of Places2 dataset among all compared models. From left to right: Masked image, DeepFill_v2, HiFill, Iconv, AOT-GAN, HiFill, CRFill, TFill, and Ours. Zoom-in for details.

- CelebA

<img src="https://i.imgur.com/hPPQQ3W.jpg" width="1000" style="zoom:100%;">

Qualitative results of CelebA dataset among all compared models. From left to right: Masked image, RW, DeepFill_v2, Iconv, AOT-GAN, CRFill, TFill, and Ours. Zoom-in for details.

## Ablation study

- Transformer and HSV loss

<div align=center>
<img src="https://i.imgur.com/DoYLVKD.png" width="410" height="150"><img src="https://imgur.com/Utxgfzs.jpg" width="410" height="150">
</div>

(left) : Ablation study label of transformer and HSV experiment.

(right) : Ablation study of color deviation on inpainted images. From left to right: Masked images, w/o TotalHSV loss, and TotalHSV loss (w/o V).

## Object removal

<div align=center>
<img src="https://i.imgur.com/IYIMow7.jpg" width="1300" height="350">
</div>

Object removal (size 256×256) results. From left to right: Original image, mask, object removal result.


## Acknowledgement
This repository utilizes the codes of following impressive repositories   
- [ZITS](https://github.com/DQiaole/ZITS_inpainting)
- [LaMa](https://github.com/saic-mdal/lama)
- [CSWin Transformer](https://github.com/microsoft/CSWin-Transformer)
- [Vision Transformer](https://github.com/google-research/vision_transformer)

---
## Contact
If you have any question, feel free to contact wiwi61666166@gmail.com

## Citation
```

@inproceedings{chen2023lightweight,
 title={Lightweight Image Inpainting By Stripe Window Transformer With Joint Attention To CNN},
 author={Chen, Bo-Wei and Liu, Tsung-Jung and Liu, Kuan-Hsien},
 booktitle={2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP)},
 pages={1--6},
 year={2023},
 organization={IEEE}
 }
