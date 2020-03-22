# FAS-SGTD

---

## Introduction

Main code of [**CVPR2020 paper "Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing"**  ](https://arxiv.org/abs/2003.08061)

## Prerequisite

* Python 3.6 (numpy, skimage, scipy)

* TensorFlow >= 1.4

* opencv2

* Pillow (PIL)

* easydict

## Train

For single-frame stage:
> cd fas_sgtd_single_frame && bash train.sh

For multi-frame stage:
> cd fas_sgtd_multi_frame && bash train.sh

## Test

We provide an example model trained for OULU-NPU Protocol 1. You can download the model at [BaiduDrive](https://pan.baidu.com/s/1sVdctfWDoUFsrXOjxr6wJw)(pwd: luik) and put them into `fas_sgtd_multi_frame/model_save/`.

> cd fas_sgtd_multi_frame && bash test.sh


## License

Code: under MIT license.

It is just for **research purpose**, and commercial use is not allowed

## Citation

If you use this code, please consider citing:

 >@inproceedings{wang2020deep,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Wang, Zezheng and Yu, Zitong and Zhao, Chenxu and Zhu, Xiangyu and Qin, Yunxiao and Zhou, Qiusheng and Zhou, Feng and Lei, Zhen},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {CVPR},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  

 >@inproceedings{yu2020searching,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {CVPR},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  


 >@inproceedings{qin2019learning,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Learning Meta Model for Zero-and Few-shot Face Anti-spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Qin, Yunxiao and Zhao, Chenxu and Zhu, Xiangyu and Wang, Zezheng and Yu, Zitong and Fu, Tianyu and Zhou, Feng and Shi, Jingping and Lei, Zhen},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {AAAI},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  