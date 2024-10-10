# Adaptive-Window-Pruning-for-Efficient-Local-Motion-Deblurring

## 📢 News
- **2024.09**
  - Release the model of LMD-ViT. 
  - Release the evaluation code.
- **2024.04** Release the blur mask annotations of the ReLoBlur dataset.
- **2024.01** Paper "Adaptive-Window-Pruning-for-Efficient-Local-Motion-Deblurring" accepted by ICLR 2024.
- **2023.10** Create this repo.

## 📷 Data
The local blur mask annotations are available at this [URL](https://drive.google.com/drive/folders/1cBhtfm7vzsyAr9D6V_LwWJma845rUSlg?usp=sharing)

## 📏 Model
The model of LMD-ViT is available at this [URL](https://drive.google.com/drive/folders/1JU2U7fxZkWzNPGhxvzhQmHDmzu0JFLqC?usp=drive_link)

## 🚀 Quick Inference
### Environment

Before inferencing LMD-ViT, please install the environment on Linux:

```
pip install -U pip
pip install -r requirements.txt
```
Creat a folder named "ckpt" and another folder named "val_data":
```
cd LMD-ViT
mkdir ckpt
mkdir val_data
```
Put the downloaded model in the "ckpt" folder.

Prepare the evaluation data to ".npy" form and put them in the "val_data" folder.

### Inference

You can evaluate the LMD-ViT by using:
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
## 📌 TODO
- [ ] Further improve the performances. 
- [ ] Release the training code.

## 🎓Citations
If our code helps your research or work, please consider citing our paper and staring this repo.
The following are BibTeX references:

```
@inproceedings{
li2024adaptive,
title={Adaptive Window Pruning for Efficient Local Motion Deblurring},
author={Haoying Li and Jixin Zhao and Shangchen Zhou and Huajun Feng and Chongyi Li and Chen Change Loy},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=hI18CDyadM}
}
```

## 🎫 License
This project and related weights are released under the [Apache 2.0 license](LICENSE).
