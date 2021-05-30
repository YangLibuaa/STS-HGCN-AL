# STS-HGCN-AL
Original PyTorch implementation of "Spatio-Temporal-Spectral Hierarchical Graph Convolutional Network With Semisupervised Active Learning for Patient-Specific Seizure Prediction" (IEEE Transactions on Cybernetics 2021).

**Paper: [https://ieeexplore.ieee.org/document/9440862](https://ieeexplore.ieee.org/document/9440862)**

![STS-HGCN-AL](https://github.com/YangLibuaa/STS-HGCN-AL/blob/main/Figures/STS-HGCN-AL.jpg)

## Requirements
The code was implemented using Python 3.8.3 and the following packages:
- torch==1.4.0
- numpy==1.18.5
- scipy==1.5.0

## Datasets
STS-HGCN with an active preictal interval learning scheme is evaluated on one public dataset with 19 patients with intractable seizures:
- [CHB-MIT dataset](http://archive.physionet.org/physiobank/database/chbmit/) 

## Main Results

![results](https://github.com/YangLibuaa/STS-HGCN-AL/blob/main/Figures/results.jpg)

## Citations
If you find the paper or this repo useful, please cite:
```
@ARTICLE{9440862,
  author={Li, Yang and Liu, Yu and Guo, Yu-Zhu and Liao, Xiao-Feng and Hu, Bin and Yu, Tao},
  journal={IEEE Transactions on Cybernetics}, 
  title={Spatio-Temporal-Spectral Hierarchical Graph Convolutional Network With Semisupervised Active Learning for Patient-Specific Seizure Prediction}, 
  year={2021},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TCYB.2021.3071860}}
```
## Contacts
For questions or help, feel welcome to write an email to [liyang@buaa.edu.cn](liyang@buaa.edu.cn) or [sy1803113@buaa.edu.cn](sy1803113@buaa.edu.cn).
