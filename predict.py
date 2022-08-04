import numpy as np
import pickle
import os
import cv2
import shutil
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
from configs.configs import (_model_path, _num_classes, _class_id_to_label_name_path, _confidence_threshold,
                            _model_name, _checkpoint_type, _hub_type, _data_dir, _output_dir)


class EfficientNetV2:
    def __init__(self, model_path: str, 
                    num_classes: int, 
                    class_id_to_label_name_path: str,
                    model_name: str, 
                    ckpt_type: str, 
                    hub_type: str):
        self.model_path = model_path
        self.num_classes = num_classes
        self.class_id_to_label_name_path = class_id_to_label_name_path
        
        self.model_name = model_name
        self.ckpt_type = ckpt_type
        self.hub_type = hub_type

        self.model = None

        with open(class_id_to_label_name_path, 'rb') as f:
            self.class_id_to_label_name = pickle.load(f)

        for class_id in self.class_id_to_label_name.keys():
            if self.class_id_to_label_name[class_id] == "unclass":
                self.class_id_to_label_name[class_id] = "undefined"

        self.hub_url, self.image_size = self.get_hub_url_and_isize(self.model_name, self.ckpt_type, self.hub_type)

    def get_class_id_to_label_name(self):
        return self.class_id_to_label_name
    
    def get_hub_url_and_isize(self, model_name, ckpt_type, hub_type):
        if ckpt_type == '1k':
            ckpt_type = ''  # json doesn't support empty string
        else:
            ckpt_type = '-' + ckpt_type  # add '-' as prefix

        hub_url_map = {
            'efficientnetv2-b0': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0/{hub_type}',
            'efficientnetv2-b1': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1/{hub_type}',
            'efficientnetv2-b2': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2/{hub_type}',
            'efficientnetv2-b3': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3/{hub_type}',
            'efficientnetv2-s':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s/{hub_type}',
            'efficientnetv2-m':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m/{hub_type}',
            'efficientnetv2-l':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/{hub_type}',

            'efficientnetv2-b0-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0-21k/{hub_type}',
            'efficientnetv2-b1-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1-21k/{hub_type}',
            'efficientnetv2-b2-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2-21k/{hub_type}',
            'efficientnetv2-b3-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3-21k/{hub_type}',
            'efficientnetv2-s-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s-21k/{hub_type}',
            'efficientnetv2-m-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m-21k/{hub_type}',
            'efficientnetv2-l-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l-21k/{hub_type}',
            'efficientnetv2-xl-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-xl-21k/{hub_type}',

            'efficientnetv2-b0-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0-21k-ft1k/{hub_type}',
            'efficientnetv2-b1-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1-21k-ft1k/{hub_type}',
            'efficientnetv2-b2-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2-21k-ft1k/{hub_type}',
            'efficientnetv2-b3-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3-21k-ft1k/{hub_type}',
            'efficientnetv2-s-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s-21k-ft1k/{hub_type}',
            'efficientnetv2-m-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m-21k-ft1k/{hub_type}',
            'efficientnetv2-l-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l-21k-ft1k/{hub_type}',
            'efficientnetv2-xl-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-xl-21k-ft1k/{hub_type}',
                
            # efficientnetv1
            'efficientnet_b0': f'https://tfhub.dev/tensorflow/efficientnet/b0/{hub_type}/1',
            'efficientnet_b1': f'https://tfhub.dev/tensorflow/efficientnet/b1/{hub_type}/1',
            'efficientnet_b2': f'https://tfhub.dev/tensorflow/efficientnet/b2/{hub_type}/1',
            'efficientnet_b3': f'https://tfhub.dev/tensorflow/efficientnet/b3/{hub_type}/1',
            'efficientnet_b4': f'https://tfhub.dev/tensorflow/efficientnet/b4/{hub_type}/1',
            'efficientnet_b5': f'https://tfhub.dev/tensorflow/efficientnet/b5/{hub_type}/1',
            'efficientnet_b6': f'https://tfhub.dev/tensorflow/efficientnet/b6/{hub_type}/1',
            'efficientnet_b7': f'https://tfhub.dev/tensorflow/efficientnet/b7/{hub_type}/1',
        }

        image_size_map = {
            'efficientnetv2-b0': 224,
            'efficientnetv2-b1': 240,
            'efficientnetv2-b2': 260,
            'efficientnetv2-b3': 300,
            'efficientnetv2-s':  384,
            'efficientnetv2-m':  480,
            'efficientnetv2-l':  480,
            'efficientnetv2-xl':  512,

            'efficientnet_b0': 224,
            'efficientnet_b1': 240,
            'efficientnet_b2': 260,
            'efficientnet_b3': 300,
            'efficientnet_b4': 380,
            'efficientnet_b5': 456,
            'efficientnet_b6': 528,
            'efficientnet_b7': 600,
        }

        hub_url = hub_url_map.get(model_name + ckpt_type)
        image_size = image_size_map.get(model_name, 224)
        return hub_url, image_size
    

    def load_model(self):
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            # Explicitly define the input shape so the model can be properly
            # loaded by the TFLiteConverter
            tf.keras.layers.InputLayer(input_shape=[self.image_size, self.image_size, 3]),
            hub.KerasLayer(self.hub_url, trainable=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(512,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                    activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(rate=0.2),
            # tf.keras.layers.Dense(256,
            #                         kernel_regularizer=tf.keras.regularizers.l2(0.00001),
            #                         activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(self.num_classes,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                    activation='softmax')
        ])
        model.build((None, self.image_size, self.image_size, 3))
        model.load_weights(self.model_path)
        self.model = model


    def classify(self, img):
        prediction_scores = self.model.predict(img)
        sorted_predicted_indexes = np.array(prediction_scores).argsort()[::-1]
        sorted_prediction_scores = [prediction_scores[0][i] for i in sorted_predicted_indexes]

        return sorted_predicted_indexes[0], sorted_prediction_scores[0]



# room_type_classification = EfficientNetV2(model_path=_model_path,
#                                     num_classes=_num_classes,
#                                     class_id_to_label_name_path=_class_id_to_label_name_path,
#                                     model_name=_model_name,
#                                     ckpt_type=_checkpoint_type,
#                                     hub_type=_hub_type)

# room_type_classification.load_model()
# class_id_to_label_name = room_type_classification.get_class_id_to_label_name()
# hub_url, image_size = room_type_classification.get_hub_url_and_isize(_model_name, _checkpoint_type, _hub_type)

# path = "/disk_local/vypham/dataset/room_type/train/loggia/3AjQfDgPSWILJUwe0IYPsOtDezgSnh98ROhxA78ZMho.jpg"
# img = cv2.imread(path)
# if img is not None:
#     img = cv2.resize(img, (image_size, image_size))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     sorted_predicted_indexes, sorted_prediction_scores = room_type_classification.classify(img)
#     for i in range(3):
#         label_name = class_id_to_label_name[sorted_predicted_indexes[-i]]
#         confidence = sorted_prediction_scores[-i]
#         print(label_name, confidence)

# aug_data = {}
# for label_name in class_id_to_label_name.values():
#     aug_data[label_name] = 0
#     if os.path.exists(os.path.join(_output_dir, label_name)) is False:
#         os.mkdir(os.path.join(_output_dir, label_name))


# for image in tqdm(os.listdir(_data_dir), desc=f"Classifying {_data_dir} dir"):
#   img = cv2.imread(os.path.join(_data_dir, image))
#   if img is not None:
#     img = cv2.resize(img, (image_size, image_size))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     sorted_predicted_indexes, sorted_prediction_scores = room_type_classification.classify(img)

#     label_name = class_id_to_label_name[sorted_predicted_indexes[-1]]
#     confidence = sorted_prediction_scores[-1]

#     print(sorted_predicted_indexes, sorted_prediction_scores)
#     if confidence > _confidence_threshold:
#         # print(f"Copy {label_name}-{confidence}")
#         shutil.copy(os.path.join(_data_dir, image), os.path.join(_output_dir,  label_name, f"{label_name}_" + "{:0.2}".format(confidence) + ".jpg"))
#         aug_data[label_name] += 1

# print("Processed done with ", aug_data)