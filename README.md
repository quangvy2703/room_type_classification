# EfficientNetV2

## 1. About EfficientNetV2 Models

[EfficientNetV2](https://arxiv.org/abs/2104.00298) are a family of image classification models, which achieve better parameter efficiency and faster training speed than prior arts.  Built upon [EfficientNetV1](https://arxiv.org/abs/1905.11946), our EfficientNetV2 models use neural architecture search (NAS) to jointly optimize model size and training speed, and are scaled up in a way for faster training and inference speed.

<img src="./g3doc/train_params.png" width="50%" />

Here are the comparison on parameters and flops:

<img src="./g3doc/param_flops.png" width="80%" />


## 2. Pretrained EfficientNetV2 Checkpoints

We have provided a list of results and checkpoints as follows:

|      ImageNet1K   |     Top1 Acc.  |    Params   |  FLOPs   | Inference Latency | links  |
|    ----------     |      ------    |    ------   | ------  | ------   |   ------   |
|    EffNetV2-S     |    83.9%   |    21.5M    |  8.4B    | [V100/A100](g3doc/effnetv2-s-gpu.png) |  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s.tgz),  [tensorboard](https://tensorboard.dev/experiment/wozwYcXkRPia76RopgCLlg)
|    EffNetV2-M     |    85.2%   |    54.1M    | 24.7B    | [V100/A100](g3doc/effnetv2-m-gpu.png) |  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz),  [tensorboard](https://tensorboard.dev/experiment/syoaqB2gTP6Vr0KRlrezmg)
|    EffNetV2-L     |    85.7%   |   119.5M    | 56.3B    | [V100/A100](g3doc/effnetv2-l-gpu.png) |  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-l.tgz),  [tensorboard](https://tensorboard.dev/experiment/qgnTQ5JZQ92nSex6ZlWBbQ)

** Thanks NVIDIA for providing the inference latency: full TensorRT scripts and instructions are available here: [link](https://github.com/NVIDIA/TensorRT/tree/master/samples/python/efficientnet)


Here are a list of ImageNet21K pretrained and finetuned models:


|  ImageNet21K  |  Pretrained models |  Finetuned ImageNet1K |
|  ----------   |  ------            |         ------       |
|  EffNetV2-S   |  [pretrain ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s-21k.tgz)  |  top1=84.9%,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s-21k-ft1k.tgz),  [tensorboard](https://tensorboard.dev/experiment/7sga2olqTBeH4ioydel0hg/) |
|  EffNetV2-M   |  [pretrain ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m-21k.tgz)  |  top1=86.2%,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m-21k-ft1k.tgz),  [tensorboard](https://tensorboard.dev/experiment/HkV6ANZSQ6WI5GhlZa48xQ/) |
|  EffNetV2-L   |  [pretrain ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-l-21k.tgz)  |  top1=86.9%,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-l-21k-ft1k.tgz),  [tensorboard](https://tensorboard.dev/experiment/m9ZHx1L6SQu5iBYhXO5jOw/) |
|  EffNetV2-XL   |  [pretrain ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-xl-21k.tgz)  |  top1=87.2%,  [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-xl-21k-ft1k.tgz),  [tensorboard]()|

For comparison with EfficientNetV1, we have also provided a few smaller V2 models using the same scaling and preprocessing as V1:

|      ImageNet1K    | Top1 Acc.  |    Params  |  FLOPs   |  links  |
|    ----------      |  ------    |    ------  | ------   |  ------   |
|    EffNetV2-B0     |    78.7%   |    7.1M    | 0.72B    | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0.tgz),  [tensorboard](https://tensorboard.dev/experiment/BbuZYLXTQXetgnrxXyAyHg/)
|    EffNetV2-B1     |    79.8%   |    8.1M    | 1.2B     | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b1.tgz),  [tensorboard](https://tensorboard.dev/experiment/2xvXQSROTZi674hjfmMXkA)
|    EffNetV2-B2     |    80.5%   |   10.1M    | 1.7B     | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b2.tgz),  [tensorboard](https://tensorboard.dev/experiment/KrdCweUDRoCkREMTJTLvuQ/)
|    EffNetV2-B3     |    82.1%   |   14.4M    | 3.0B     | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b3.tgz),  [tensorboard](https://tensorboard.dev/experiment/0nYo4rMDTQuQcqOFzUMddA/)

Here are the ImageNet21k checkpoints and finetuned models for B0-B3:

* EffNetV2-B0: [ImageNet21k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0-21k.tgz), [ImageNet21k-ft1k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0-21k-ft1k.tgz)
* EffNetV2-B1: [ImageNet21k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b1-21k.tgz), [ImageNet21k-ft1k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b1-21k-ft1k.tgz)
* EffNetV2-B2: [ImageNet21k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b2-21k.tgz), [ImageNet21k-ft1k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b2-21k-ft1k.tgz)
* EffNetV2-B3: [ImageNet21k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b3-21k.tgz), [ImageNet21k-ft1k](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b3-21k-ft1k.tgz)

## 3. Training & Finetuning

Config the training hyper-parameters in

    configs/configs.py
|  Variable  |  Detail  |  Example |
|  ----------   |  ------            |         ------       |
|  TRAIN_DIR    |  Training images directory        | TRAIN_DIR = '/disk_local/vypham/dataset/room_type/refines/refine_6/train/' |
|  VALID_DIR    |  Validating images directory      | VALID_DIR = '/disk_local/vypham/dataset/room_type/refines/refine_6/valid/' |
|  _training    |  Train or Validate                | _training = True |
|  _train_both  |  Train both train and valid data  | _train_both = True |
|  _model_path  |  Pretrained model path            | _model_path = "/disk_local/vypham/room_type_classification/models/efficientnetv2-b1/refine_6_both_mdl_wts.hdf5" |
|  _num_classes |  Num of classes                   | _num_classes = 17 |
|  _batch_size  |  Batch size                       | _batch_size = 64 |
|  _num_classes |  Num of classes                   | _num_classes = 17 |
|  _batch_size  |  Batch size                       | _epochs = 100 |
|  _epochs      |  Num of training epochs           | _num_classes = 17 |
|  _class_id_to_label_name_path  |  Mapping file from class id to label name | _class_id_to_label_name_path = "/disk_local/vypham/room_type_classification/class_id_to_label_name.pkl" |
|  _model_name  |  Using model                      | _model_name = "efficientnetv2-b1" |
|  _options     |  Suffix saved model name          | _options = "refine_6_both" |



Train room-type classification:

    python train_effv2.py
    # Trained model will be saved at models/{model_name}/{options}_mdl_wts.hdf5
    # Set _train_both=True and _epochs to epoch model start convergin to train the deploy model

Test room-type classification:

    # Change the config _training=False in config/configs.py
    python train.py
    # The results will be saved at logs/{model_name}/{model_name}_{options}.csv

Inference room-type classification:
    img = cv2.imread(os.path.join(trainer.data_dir, 'train', image_class, image))
    img = cv2.resize(img, (trainer.model.image_size, trainer.model.image_size), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    sorted_predicted_indexes, sorted_prediction_scores = trainer.model.classify(img)

    label_name = trainer.model.class_id_to_label_name[sorted_predicted_indexes[-1]]
    score = sorted_prediction_scores[-1]

Refine original room-type dataset:

    # Some problems with original dataset
        1. Dining room view contains Kitchen room view
        2. Dining room view contains Living room view
        3. Toilet room view contains Bathroom view
    In order to surpass this problems, I propose to add some multi-view classes
        1. Dining-Kitchen room view
        2. Dining-Living room view
        3. Toilet-Bathroom view
    Base on the furniture detection, the multi-view defined as below:
            kitchen = ["below_kitchen_cabinet", "oven", "gas_stove", "above_kitchen_cabinet", "kitchen_hood", "microwave", "sink"]
            dining_room = ["dining_table_and_chairs"]
            bathroom = ["shower_glass", "shower", "bathtub", "basin"]
            toilet = ["toilet_bowl"]
            living_room = ["sofa", "tv"]
            bedroom = ['bedside_cupboard', 'bed', 'makeup_table']
        * Images with view contains kitchen and dining view will be classed into Dining-Kitchen room
        * Images with view contains dining and living room view will be classed into Dining-Living room
        * Images with view contains toilet and bathroom view will be classed into Toilet-Bathroom room
    In order to refine, get the furniture detection repo and run:
        python detect.py --weights /disk_local/vypham/yolov5/runs/train/yolov5m_refine_0/weights/best.pt --img 512 --conf 0.3 --source ../{PATH_TO_ALL_IMAGE_DIR} --refine_room_type True
    Follow the code in detect.py to get the refined classes
    Finally, copy remain classes to the refined dir


    python main.py --mode=train  --model_name=efficientnetv2-s  --dataset_cfg=cifar10Ft --model_dir=$DIR --hparam_str="train.ft_init_ckpt=$PRETRAIN_CKPT_PATH"


## 4. Build a pretrained model and finetuning


You can directly use this code to build a model like this:

    mode = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
        effnetv2_model.get_model('efficientnetv2-b0', include_top=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])

Or you can also load them from tfhub:

    hub_url = 'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0/feature-vector'
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
        hub.KerasLayer(hub_url, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])

