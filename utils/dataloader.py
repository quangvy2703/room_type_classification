import os
import sys
import shutil
from tqdm import tqdm
import random
import numpy as np
import yaml
import cv2
sys.path.insert(0, 'utils')
from models import EfficientNetV2

with open("configs/train_configs.yaml", "r") as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def train_valid_split(valid_split=0.2):
    train_classes = os.listdir(configs['TRAIN_DIR'])

    for train_class in tqdm(train_classes):
        if os.path.exists(os.path.join(configs['VALID_DIR'], train_class)) is False:
            os.system(f"mkdir {os.path.join(configs['VALID_DIR'], train_class)}")
        images = os.listdir(os.path.join(configs['TRAIN_DIR'], train_class))
        valid_images = random.sample(images, k=int(len(images) * valid_split))
        for valid_image in valid_images:
            shutil.move(os.path.join(configs['TRAIN_DIR'], train_class, valid_image), os.path.join(configs['VALID_DIR'], train_class, valid_image))
    train_images = {}
    valid_images = {}
    for train_class in tqdm(train_classes):
        train_images[train_class] = len(os.listdir(os.path.join(configs['TRAIN_DIR'], train_class)))
        valid_images[train_class] = len(os.listdir(os.path.join(configs['VALID_DIR'], train_class)))
    print(f"Splited with train \n{train_images} \n valid \n {valid_images}")

def train_valid_unsplit(train_dir, valid_dir):
    train_classes = os.listdir(train_dir)
    
    for train_class in tqdm(train_classes):
        images = os.listdir(os.path.join(valid_dir, train_class))
        for valid_image in images:
            shutil.move(os.path.join(valid_dir, train_class, valid_image), os.path.join(train_dir, train_class, valid_image))
    train_images = {}
    valid_images = {}
    for train_class in tqdm(train_classes):
        train_images[train_class] = len(os.listdir(os.path.join(train_dir, train_class)))
        valid_images[train_class] = len(os.listdir(os.path.join(valid_dir, train_class)))
    print(f"Splited with train \n{train_images} \n valid \n {valid_images}")

def merge_dataset(main_data_path, aug_data_path):
    # "walkin_closet", "tabernacle_room", "laundry_room", "working_room", "hallway", "stairs", 
    aug_classes = os.listdir(aug_data_path)
    before_merge = {}
    for aug_class in aug_classes:
        before_merge[aug_class] = len(os.listdir(os.path.join(main_data_path, aug_class)))

    for aug_class in aug_classes:
        # shutil.copytree(os.path.join(aug_data_path, aug_class), os.path.join(main_data_path, aug_class))
        os.system(f"cp {os.path.join(aug_data_path, aug_class)}/* {os.path.join(main_data_path, aug_class)}")

    after_merge = {}
    for aug_class in aug_classes:
        after_merge[aug_class] = len(os.listdir(os.path.join(main_data_path, aug_class)))
    print("Before merge, ", before_merge)
    print("After merge, ", after_merge)
    

def unmerge_dataset(main_data_path, aug_data_path):
    # aug_classes = ["walkin_closet", "tabernacle_room", "laundry_room", "working_room", "hallway", "stairs", "undefined"]
    aug_classes = os.listdir(aug_data_path)
    # aug_classes = ["stairs"]
    before_unmerge = {}
    for aug_class in aug_classes:
        before_unmerge[aug_class] = len(os.listdir(os.path.join(main_data_path, aug_class)))

    for aug_class in aug_classes:
        aug_images = os.listdir(os.path.join(aug_data_path, aug_class))
        for aug_image in aug_images:
            if os.path.exists(os.path.join(main_data_path, aug_class, aug_image)):
                os.remove(os.path.join(main_data_path, aug_class, aug_image))

    after_unmerge = {}
    for aug_class in aug_classes:
        after_unmerge[aug_class] = len(os.listdir(os.path.join(main_data_path, aug_class)))
    print("Before unmerge, ", before_unmerge)
    print("After unmerge, ", after_unmerge)

def convert_gray_scale(data_path, out_dir):
    os.system(f"mkdir {data_path}/../{out_dir}")
    labels = os.listdir(data_path)
    for label in tqdm(labels):
        os.system(f"mkdir {data_path}/../{out_dir}/{label}")
        images = os.listdir(os.path.join(data_path, label))
        for image in images:
            img = cv2.imread(os.path.join(data_path, label, image), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cv2.imwrite(f"{data_path}/../{out_dir}/{label}/{image}", img)
