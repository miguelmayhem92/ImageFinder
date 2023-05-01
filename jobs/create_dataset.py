import pandas as pd
import numpy as np

import os
import random

import shutil

from configs import configs

my_local_path = os.getcwd()

target_file = configs.csv_file 
clean_data_name = configs.clean_data_interview
n_train_images = configs.n_train_images
seed_data = configs.seed_data

csv_path = my_local_path + f"/{target_file}{clean_data_name}"

image_id = pd.read_csv(csv_path, sep = ',', error_bad_lines=False)

orders = list(range(len(image_id)))
random.seed(seed_data)
random.shuffle(orders)

## type creates a label train/test for the images inside the csv 
image_id['order'] = orders
image_id['type'] = np.where(image_id.order <= n_train_images, 'train', 'test')

image_id = image_id.drop(columns = ['order'])

## the following loop copies the image from the image/ file and paste it to the dataset/train or dataset/test
for i in range(len(image_id)):
    row = image_id.iloc[i]
    jpgName = row.jpgName
    typex = row.type
    
    src = my_local_path + '/images/' + jpgName
    dst = my_local_path + f'/dataset/{typex}/' + jpgName
    
    if not os.path.exists(my_local_path + f'/dataset/{typex}/'):
        os.makedirs(my_local_path + f'/dataset/{typex}/')
    
    shutil.copyfile(src, dst)

## this creates a csv file for each train/ and test/ file that contains metadata for each image (needed to create dataset_object in the train step) 
def create_metadata(data,type_x):
    tmp = data[data.type == type_x][['jpgName','id','url']]
    tmp['file_name'] = tmp.jpgName
    tmp = tmp[['file_name','jpgName','id','url']]
    tmp.to_csv(my_local_path + f'/dataset/{type_x}/' + 'metadata.csv', index = False, header = True )
    return print(f'save {type_x}')

create_metadata(image_id,'train')
create_metadata(image_id,'test')