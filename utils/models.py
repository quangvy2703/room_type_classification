import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow_hub as hub

print('TF version:', tf.__version__)
print('Hub version:', hub.__version__)
print('Phsical devices:', tf.config.list_physical_devices())

class EfficientNetV2:
  def __init__(self, model_name, pretrained_path, efficient_path, batch_size=64, train_data_dir=None, \
              valid_data_dir=None, do_data_augmentation=None, epochs=None, class_id_to_label_name_path=None, options=None,
              num_classes=None) -> None:
    self.model_name = model_name
    self.pretrained_path = pretrained_path
    self.efficient_path = efficient_path
    self.batch_size = batch_size
    self.train_data_dir = train_data_dir
    self.valid_data_dir = valid_data_dir
    self.epochs = epochs
    self.class_id_to_label_name_path = class_id_to_label_name_path
    self.options = options
    self.num_classes = num_classes
    self.ckpt_type = '1k'   # @param ['21k', '21k-ft1k', '1k']
    self.hub_type = 'feature-vector' # @param ['feature-vector']
    self.do_data_augmentation = do_data_augmentation #@param {type:"boolean"}
    self.get_image_size(model_name)
    pass
  
  
  def get_image_size(self, model_name):   
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
    self.image_size = image_size_map.get(model_name, 224)
    return self.image_size

  def setup_dirs(self):
    if os.path.exists(f'models/{self.model_name}') is False:
      os.mkdir(f'models/{self.model_name}')
    if os.path.exists(f"logs/{self.model_name}") is False:
      os.mkdir(f"logs/{self.model_name}")
    # if os.path.join(f"results/{self.model_name}") is False:
    #   os.mkdir(f"results/{self.model_name}")
    
  def save_mapping_file(self):
    class_id_to_label_name = dict([(v, k) for k, v in self.valid_generator.class_indices.items()])
    import pickle
    with open(self.class_id_to_label_name_path, 'wb') as f:
      pickle.dump(class_id_to_label_name, f)

  def load_mapping_file(self):
    import pickle
    with open(self.class_id_to_label_name_path, 'rb') as f:
      self.class_id_to_label_name = pickle.load(f)
    

  def get_random_eraser(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

  def data_loader(self):
    self.image_size = self.get_image_size(self.model_name)
    self.load_mapping_file()
    datagen_kwargs = dict(rescale=1./255)

    dataflow_kwargs = dict(target_size=(self.image_size, self.image_size),
                          batch_size=self.batch_size,
                          interpolation="bilinear",
                          seed=1)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    self.valid_generator = valid_datagen.flow_from_directory(
        self.valid_data_dir, subset="training", shuffle=False, **dataflow_kwargs)

    
    if self.do_data_augmentation:
      train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=40,
          horizontal_flip=True,
          width_shift_range=0.2, height_shift_range=0.2,
          shear_range=0.2, zoom_range=0.2,
          preprocessing_function=self.get_random_eraser(v_l=0, v_h=255),
          **datagen_kwargs)
    else:
      train_datagen = valid_datagen
    self.train_generator = train_datagen.flow_from_directory(
        self.train_data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    self.save_mapping_file()

  def model_loader(self):
    tf.keras.backend.clear_session()
    self.model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[self.image_size, self.image_size, 3]),
        hub.KerasLayer(self.efficient_path, trainable=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(512,
                              kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                              activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(self.num_classes,
                              kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                              activation='softmax')
    ])
    self.model.build((None, self.image_size, self.image_size, 3))
    if self.pretrained_path is not None:
      self.model.load_weights(self.pretrained_path)
      self.load_mapping_file()
    self.model.summary()

  def model_fit(self):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(f'models/{self.model_name}/{self.options}_mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
    logger = CSVLogger(f"logs/{self.model_name}/{self.options}.csv", separator=',', append=True)

      
    self.model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
      metrics=['accuracy'])


    steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

    hist = self.model.fit(
        self.train_generator,
        epochs=self.epochs, steps_per_epoch=steps_per_epoch,
        validation_data=self.valid_generator,
        validation_steps=validation_steps,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss, logger]).history
  
  
  def classify(self, img):
      prediction_scores = self.model.predict(img)
      sorted_predicted_indexes = np.array(prediction_scores).argsort()[::-1]
      sorted_prediction_scores = [prediction_scores[0][i] for i in sorted_predicted_indexes]
      return sorted_predicted_indexes[0], sorted_prediction_scores[0]
      

  def accuracy_calculate(self, y_pred, y):
    accuracy = {}
    overall = 0
    for _class in self.class_id_to_label_name.keys():
      accuracy[_class] = 0
    for _y_pred, _y in zip(y_pred, y):
      accuracy[_y] = accuracy[_y] + 1 if _y_pred == _y else accuracy[_y]
      overall = overall + 1 if _y_pred == _y else overall
    results = {}
    _, counts = np.unique(y, return_counts=True)
    for i in accuracy.keys():
      class_name = self.class_id_to_label_name[i]
      results[class_name] = accuracy[i] / counts[i]
    overall = overall / len(y_pred)
    return results, overall

  def model_test(self):
    from sklearn.metrics import classification_report
    from tqdm import tqdm
    validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
    predicted = []
    ground_truth = []
    for i in tqdm(range(validation_steps)):
      batch = next(self.valid_generator)
      imgs, labels = batch[0], batch[1]
      ground_truth.extend(np.argmax(labels, axis=1))
      predicted.extend(np.argmax(self.model.predict(imgs), axis=1))
    report = classification_report(ground_truth, predicted, output_dict=True, target_names=self.class_id_to_label_name.values())
    accuracy, overall = self.accuracy_calculate(predicted, ground_truth)
    results = {}
    for _class in accuracy.keys():
      results[_class] = {
        'Precision': report[_class]['precision'],
        'Recall': report[_class]['recall'],
        'F1-Score': report[_class]['f1-score'],
        'Accuracy': accuracy[_class],
        'Accuracy_Overall': overall
      }

    pd.DataFrame(results).to_csv(
        f"logs/{self.model_name}/{self.model_name}_{self.options}.csv",
        mode='a' if os.path.exists(f"logs/{self.model_name}/{self.model_name}_{self.options}.csv") else 'w', 
        index=True,
        header=False if os.path.exists(f"logs/{self.model_name}/{self.model_name}_{self.options}.csv") else True
    )


# O: Mục tiêu hướng tới là đưa công ty phát triển mạnh hơn về mặt công nghệ và kỹ thuật dựa vào trí tuệ nhân tạo, đẩy nhanh các quy trình xử lí dựa vào trí tuệ nhân tạo.
# KR1: Tỷ lệ phát sinh lỗi trong quá trình sử dụng giảm còn 1% đến 3%
# KR2: Tối ưu thời gian thực thi của hệ thống 40-50%
# KR3: Hiệu suất làm việc của nhân viên tăng 40% thông qua hỗ trợ của trí tuệ nhân tạo