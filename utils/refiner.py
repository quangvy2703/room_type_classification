import os
import sys
import yaml
import cv2
import numpy as np
from tqdm import tqdm
sys.path.insert(0, 'utils')
from models import EfficientNetV2

with open("configs/train_configs.yaml", "r") as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class Refine():
    def __init__(self, n_loops, root_dir, data_dir, output_dir, swap_classes, confidence_threshold, model_configs) -> None:
        self.n_loops = n_loops
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.swap_classes = swap_classes
        self.confidence_threshold = confidence_threshold
        self.model_configs = model_configs
        self.model = EfficientNetV2(self.model_configs['model_name'], self.model_configs['model_path'],  
                                        self.model_configs['efficient_path'],self.model_configs['batch_size'], 
                                        self.model_configs['train_data_dir'], self.model_configs['valid_data_dir'], 
                                        self.model_configs['use_augmentation'], self.model_configs['epochs'], 
                                        self.model_configs['class_id_to_label_name_path'], self.model_configs['options'],
                                        self.model_configs['num_classes'])

        self.save_file = os.path.join(self.output_dir, "refines.csv") 

    def refine(self, i_loop):
        results = {}
        results['model_name'] = configs['model_name']

        if os.path.exists(os.path.join(self.output_dir)) is False:
            os.mkdir(os.path.join(self.output_dir))

        if os.path.exists(os.path.join(self.output_dir, 'train')) is False:
            os.mkdir(os.path.join(self.output_dir, 'train'))
            print("Created ", os.path.join(self.output_dir + '/train'))

        for i in self.swap_classes:
            if os.path.exists(os.path.join(self.output_dir, 'train', i)) is False:
                os.mkdir(os.path.join(self.output_dir, 'train', i))

        for image_class in tqdm(os.listdir(self.data_dir + '/train'), desc=f"Classifying {self.data_dir + '/train'} dir"):
            # if os.path.exists(os.path.join(self.output_dir, image_class)) is False:
            #     os.mkdir(os.path.join(self.output_dir, image_class))


            if image_class not in self.swap_classes:
                os.system(f"cp -a {self.data_dir}/train/{image_class}/ {self.output_dir}/train/{image_class}/")
                print(f"cp -a {self.data_dir}/train/{image_class}/ {self.output_dir}/train/{image_class}/")
                continue

            swap_images = {}
            swap_images[image_class] = []
            for image in tqdm(os.listdir(os.path.join(self.data_dir, 'train', image_class))):
                img = cv2.imread(os.path.join(self.data_dir, 'train', image_class, image))
                if img is not None:
                    img = cv2.resize(img, (self.model.image_size, self.model.image_size), interpolation=cv2.INTER_LINEAR)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    sorted_predicted_indexes, sorted_prediction_scores = self.model.classify(img)

                    label_name = self.model.class_id_to_label_name[sorted_predicted_indexes[-1]]
                    score = sorted_prediction_scores[-1]

                    if label_name in self.swap_classes and score > self.confidence_threshold:
                        os.system(f"cp \"{os.path.join(self.data_dir, 'train', image_class, image)}\" {os.path.join(self.output_dir, 'train', label_name)}") 
                    else:
                        os.system(f"cp \"{os.path.join(self.data_dir, 'train', image_class, image)}\" {os.path.join(self.output_dir, 'train', image_class)}") 


   
        if os.path.exists(self.output_dir + '/valid') is False:
            os.mkdir(self.output_dir + '/valid')
            print("Created ", os.path.join(self.output_dir + '/valid'))

        for i in self.swap_classes:
            if os.path.exists(os.path.join(self.output_dir, 'valid', i)) is False:
                os.mkdir(os.path.join(self.output_dir, 'valid', i))
        for image_class in tqdm(os.listdir(self.data_dir + '/valid'), desc=f"Classifying {self.data_dir + '/valid'} dir"):
            # if os.path.exists(os.path.join(self.output_dir, image_class)) is False:
            #     os.mkdir(os.path.join(self.output_dir, image_class))


            if image_class not in self.swap_classes:
                os.system(f"cp -a {self.model_configs['valid_data_dir']}/{image_class}/ {self.output_dir}/valid/{image_class}/")
                continue

            swap_images = {}
            swap_images[image_class] = []
            for image in tqdm(os.listdir(os.path.join(self.data_dir, 'valid', image_class))):
                img = cv2.imread(os.path.join(self.data_dir, 'valid', image_class, image))
                if img is not None:
                    img = cv2.resize(img, (self.model.image_size, self.model.image_size), interpolation=cv2.INTER_LINEAR)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    sorted_predicted_indexes, sorted_prediction_scores = self.model.classify(img)

                    label_name = self.model.class_id_to_label_name[sorted_predicted_indexes[-1]]
                    score = sorted_prediction_scores[-1]

                    if label_name in self.swap_classes and score > self.confidence_threshold:
                        os.system(f"cp \"{os.path.join(self.data_dir, 'valid', image_class, image)}\" {os.path.join(self.output_dir, 'valid', label_name)}") 
                    else:
                        os.system(f"cp \"{os.path.join(self.data_dir, 'valid', image_class, image)}\" {os.path.join(self.output_dir, 'valid', image_class)}") 
        
        

    def refine_loop(self):
        for i_loop in range(0, self.n_loops + 10):          
            self.model.options = self.model_configs['options'].split("_")[0] + f'_{i_loop}'
            self.data_dir = os.path.join(self.root_dir, f'refine_{i_loop}')
            self.output_dir = os.path.join(self.root_dir, f'refine_{i_loop + 1}')
            self.model_configs['options'] = f'refine_{i_loop}'
            # At the first loop, we use the original dataset
            if i_loop > 0:
                self.model.train_data_dir = self.data_dir + '/train'
                self.model.valid_data_dir = self.data_dir + '/valid'   
            

            self.model.setup_dirs()
            self.model.data_loader()
            if i_loop > 0:
                self.model.pretrained_path = f"/disk_local/vypham/room_type_classification/models/efficientnetv2-b1/refine_{i_loop - 1}_mdl_wts.hdf5"
            self.model.model_loader()
            self.model.model_fit()
            self.refine(i_loop)
