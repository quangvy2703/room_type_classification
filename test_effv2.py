
import os
import shutil
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from utils.models import EfficientNetV2
import yaml
import pandas as pd

def main():
    with open("configs/valid_configs.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    evaluator = EfficientNetV2(
		model_name=configs['model_name'], 
		pretrained_path=configs['model_path'], 
		efficient_path=configs['efficient_path'], 
		class_id_to_label_name_path=configs['class_id_to_label_name_path'],
        num_classes=configs['num_classes']
  	)

    evaluator.model_loader()
    evaluator.load_mapping_file()
    image_size = evaluator.get_image_size(configs['model_name'])

    datagen_kwargs = dict(rescale=1./255)

    dataflow_kwargs = dict(target_size=(image_size, image_size),
                        batch_size=64,
                        interpolation="bilinear",
                        seed=1)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        configs['VALID_DIR'], subset="training", shuffle=False, **dataflow_kwargs)


    results = {}
    save_dir = "/disk_local/vypham/room_type_classification/results_b1/"

    tp = 0
    n_samples = 0
    results['model_name'] = configs['model_name']
    results['batch norm'] = [True]
    for image_classes in tqdm(os.listdir(configs['VALID_DIR']), desc=f"Classifying {configs['VALID_DIR']} dir"):
        if os.path.exists(os.path.join(save_dir, image_classes)) is False and configs['save_image_results']:
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)
            os.mkdir(save_dir + image_classes)
        results[image_classes] = 0
        for image in tqdm(os.listdir(os.path.join(configs['VALID_DIR'], image_classes))):
            img = cv2.imread(os.path.join(configs['VALID_DIR'], image_classes, image))
            if img is not None:
                n_samples += 1
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                sorted_predicted_indexes, sorted_prediction_scores = evaluator.classify(img)

                label_name = evaluator.class_id_to_label_name[sorted_predicted_indexes[-1]]
                score = sorted_prediction_scores[-1]

                label_name_2 = evaluator.class_id_to_label_name[sorted_predicted_indexes[-2]]
                score_2 = sorted_prediction_scores[-2]

                if label_name == image_classes:
                    results[image_classes] += 1     
                    tp += 1      
                
                if configs['save_image_results']:
                    shutil.copy(os.path.join(configs['VALID_DIR'], image_classes, image), os.path.join(save_dir, image_classes, f'{label_name}_' + "{:2.2f}_".format(score) + f'{label_name_2}_' + "{:2.2f}_".format(score_2) + image))

        total = len(os.listdir(os.path.join(configs['VALID_DIR'], image_classes)))
        results[image_classes] = [100 * results[image_classes] / total ]

    results['overall'] = [100 * tp / n_samples]


    pd.DataFrame(results).to_csv(
        configs['save_file'],
        mode='w', 
        index=True,
        header=True
    )

    print(f"Evaluate done with classification results saved in {configs['save_file']}")
    

if __name__ == '__main__':
    main()