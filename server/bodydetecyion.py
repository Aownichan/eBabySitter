import tensorflow as tf
import os

import cv2
import imghdr

data_dir = 'babies' 
image_exts = ['jpeg','jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir): 

    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('babies')

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()