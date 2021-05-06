import os
import sys

from utils.dataset_utils import get_classes_num

train_images_path = "images/train"
test_images_path = "images/test"

train_data_path = 'train_data.csv'
test_data_path = 'test_data.csv'

model_path = "models"

img_height = 8
img_width = 8

batch_size = 1

keras_epochs = 3
pnet_epochs = 3

classes_num = get_classes_num(path=train_images_path)
	
