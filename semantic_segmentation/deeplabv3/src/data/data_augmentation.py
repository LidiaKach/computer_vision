import tensorflow as tf
from PIL import Image
import numpy as np

from src.config import IMG_SIZE


def aug_img_mask(image, mask, i=0, augment=True):
    mask = tf.expand_dims(mask, axis=2)

    # resize image and mask
    input_image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')

    # # rescale the image
    # input_image = tf.cast(input_image, tf.float32) / 255.0

    # augmentation
    if augment:
        # zoom in a bit
        if i == 0:
            input_image = tf.image.central_crop(input_image, 0.75)
            input_mask = tf.image.central_crop(input_mask, 0.75)
            # resize
            input_image = tf.image.resize(input_image, (IMG_SIZE, IMG_SIZE))
            input_mask = tf.image.resize(input_mask, (IMG_SIZE, IMG_SIZE), method='nearest')

        elif i == 1:
            # zoom in a bit
            input_image = tf.image.central_crop(input_image, 0.75)
            input_mask = tf.image.central_crop(input_mask, 0.75)
            # resize
            input_image = tf.image.resize(input_image, (IMG_SIZE, IMG_SIZE))
            input_mask = tf.image.resize(input_mask, (IMG_SIZE, IMG_SIZE), method='nearest')
            # flipping random horizontal or vertical
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
        elif i == 2:
            # random contrast adjustment
            input_image = tf.image.random_contrast(input_image, 0.2, 0.3)
        elif i == 3:
            input_image = tf.image.random_saturation(input_image, 1.75, 2.25)
        else:
            # flipping random horizontal
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

    return input_image.numpy(), input_mask.numpy().reshape(IMG_SIZE, IMG_SIZE)
