# Adaptive-Window-Pruning-for-Efficient-Local-Motion-Deblurring

## 📢 News
- **2024.09**
  - Release the model of LMD-ViT. 
  - Release the evaluation code.
- **2024.04** Release the blur mask annotations of the ReLoBlur dataset.
- **2024.01** Paper "Adaptive-Window-Pruning-for-Efficient-Local-Motion-Deblurring" accepted by ICLR 2024.
- **2023.10** Create this repo.

## Data
The local blur mask annotations are available at https://drive.google.com/drive/folders/1cBhtfm7vzsyAr9D6V_LwWJma845rUSlg?usp=sharing

## Model
The model of LMD-ViT is available at https://drive.google.com/drive/folders/1JU2U7fxZkWzNPGhxvzhQmHDmzu0JFLqC?usp=drive_link
## Quick Inference
### Environment

Before inferencing LMD-ViT, please install the environment on Linux:

```
pip install -U pip
pip install -r requirements.txt
```

### Inference

You can evaluate the LMD-ViT by using:
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
## 📌 TODO
- [ ] Further improve the performances. 
- [ ] Release the training code.
