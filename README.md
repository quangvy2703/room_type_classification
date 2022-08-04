# Official YOLOv7

``` bash
Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

Author's github page https://github.com/WongKinYiu/yolov7
```

---
## NOTE
``` bash
Following the instruction or got pain
```
---

## Installation

``` bash
 # Install virtualenv library
pip install virtualenv
 # Create virtual environment
python3.8 -m virtualenv yolov7_env
 # Activate virtual environment
source yolov7_env/bin/activate
 # Install required libraries to the created virtual environment
pip install -r requirements.txt
```
## Setup training data

``` bash
1. Your data must be organizated like following
root_dir
├── train
│       ├── class_0
│       │   ├── image_class_0_0.jpg
│       │   ├── image_class_0_1.jpg
│       │   ├── ...
│       │   
│       ├── class_1
│           ├── image_class_1_0.jpg
│           ├── image_class_1_0.jpg
│           ├── ...
├── valid
│       ├── class_0
│       │   ├── image_class_0_0.jpg
│       │   ├── image_class_0_1.jpg
│       │   ├── ...
│       │   
│       ├── class_1
│           ├── image_class_1_0.jpg
│           ├── image_class_1_0.jpg
│           ├── ...

2. Change the configs in config file configs/train_configs.yaml
       TRAIN_DIR: path to the training directory
       VALID_DIR: path to the validation directory 
       only_train: train without evaluate classification results
       train_both: in the case you haved estimated the converage epoch, set True to train on both training set and validation set
       save_file: path to the csv classification results 
       model_path: pretrained model path
       efficient_path: pretrained model of efficient net
       num_classes: number of classes
       batch_size: batch size
       epochs: num of training epochs
       class_id_to_label_name_path: path to mapping file from class id to label name
       model_name: selected model 
       confidence_threshold: confidence of classification
       options: surfix to the saved dir
       use_augmentation: Use the augmentation data

3. Train with train_effv2.py
```

## Training

``` bash

python train_effv2.py
```

## Inference

``` bash
python test_effv2.py
```


<div align="center">
    <a href="./">
        <img src="./figure/watermarks.jpg" width="59%"/>
    </a>
</div>
