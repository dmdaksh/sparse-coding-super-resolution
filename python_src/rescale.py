import numpy as np 
from skimage.transform import rescale
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from os import listdir
from tqdm import tqdm

# Set train and val HR and LR paths
train_hr_path = 'data/train_hr/'
train_lr_path = 'data/train_lr/'
val_hr_path = 'data/val_hr/'
val_lr_path = 'data/val_lr/'

numberOfImagesTrainHR = len(listdir(train_hr_path))

for i in tqdm(range(numberOfImagesTrainHR)):
    img_name = listdir(train_hr_path)[i]
    img = imread('{}{}'.format(train_hr_path, img_name))
    new_img = rescale(img, (1/3), anti_aliasing=1, channel_axis=2)
    if len(new_img.shape) == 2 or (len(new_img.shape) == 3 and new_img.shape[2] == 1):
        new_img = gray2rgb(new_img)
    new_img = (new_img * 255).astype(np.uint8)
    imsave('{}{}'.format(train_lr_path, img_name), new_img, quality=100)

numberOfImagesValHR = len(listdir(val_hr_path))

for i in tqdm(range(numberOfImagesValHR)):
    img_name = listdir(val_hr_path)[i]
    img = imread('{}{}'.format(val_hr_path, img_name))
    new_img = rescale(img, (1/3), anti_aliasing=1, channel_axis=2)
    if len(new_img.shape) == 2 or (len(new_img.shape) == 3 and new_img.shape[2] == 1):
        new_img = gray2rgb(new_img)
    new_img = (new_img * 255).astype(np.uint8)
    imsave('{}{}'.format(val_lr_path, img_name), new_img, quality=100)