from utils.dataloader import Refine
import yaml

n_loops = 5
root_dir = "/disk_local/vypham/dataset/room_type/refines_test"
data_dir = "/disk_local/vypham/dataset/room_type/refines_test"
output_dir = "/disk_local/vypham/dataset/room_type/refines"
swap_classes = ['bathroom', 'toilet', 'loggia', 'balcony', 'kitchen', 'dining_room']
confidence = 0.8
model_configs = {
    'train_data_dir': '/disk_local/vypham/dataset/room_type/refines/refine_6/train/',
    'valid_data_dir': '/disk_local/vypham/dataset/room_type/refines/refine_6/valid/',
    'save_file': "/disk_local/vypham/room_type_classification/results/classification_results.csv",
    'model_path': "/disk_local/vypham/room_type_classification/models/efficientnetv2-b1/refine_5_mdl_wts.hdf5",
    'efficient_path': "/disk_local/vypham/room_type_classification/data/feature-vector",
    'num_classes': 20,
    'batch_size': 64,
    'epochs': 200,
    'class_id_to_label_name_path': "/disk_local/vypham/room_type_classification/class_id_to_label_name.pkl",
    'model_name': "efficientnetv2-b1",
    'confidence_threshold': 0.8,
    'options': "refine_test_0",
    'use_augmentation': True
}
refine = Refine(n_loops, root_dir, data_dir, output_dir, swap_classes, confidence, model_configs)
refine.refine_loop()


