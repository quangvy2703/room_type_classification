import argparse
from utils.models import EfficientNetV2
import yaml
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from utils.dataloader import merge_dataset, unmerge_dataset


def main():

	with open("configs/train_configs.yaml", "r") as stream:
		try:
			configs = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	trainer = EfficientNetV2(
		model_name=configs['model_name'], 
		pretrained_path=configs['model_path'], 
		efficient_path=configs['efficient_path'], 
		batch_size=configs['batch_size'], 
		train_data_dir=configs['TRAIN_DIR'],
		valid_data_dir=configs['VALID_DIR'], 
		do_data_augmentation=configs['use_augmentation'], 
		epochs=configs['epochs'],
		class_id_to_label_name_path=configs['class_id_to_label_name_path'],
		options=configs['options'],
		num_classes=configs['num_classes']
  	)

	trainer.setup_dirs()

	if configs['train_both']:
		merge_dataset(configs['TRAIN_DIR'], configs['VALID_DIR'])
	if configs['train_both'] is False:
		unmerge_dataset(configs['TRAIN_DIR'], configs['VALID_DIR'])

	trainer.data_loader()
	trainer.model_loader()
	trainer.model_fit()
	if configs['only_train'] is False:
		trainer.model_test()

if __name__ == '__main__':
	main()
