import os
import tensorflow as tf
from PIL import Image
from src.config import IMG_SIZE
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_dir, label_dir=None, depth=None, batch_size=32, fit=True):
        self.names = os.listdir(img_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.fit = fit
        self.depth = depth

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(len(self.names) / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        if self.fit:
            batch_x = self.__get_input(idx)
            batch_y, batch_weights = self.__get_output(idx)
            return batch_x, batch_y, batch_weights
        else:
            batch_x = self.__get_input(idx)
            return batch_x

    def __get_input(self, idx):
        X = []
        idxs = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        for index in idxs:
            img_path = os.path.join(self.img_dir, index)
            img = Image.open(img_path)
            if not self.fit:
                img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            image = np.array(img) / 255.
            X.append(image)

        return np.array(X, dtype="float32")

    def __get_output(self, idx):
        y = []
        sample_weights = []
        idxs = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        for index in idxs:
            label_path = os.path.join(self.label_dir, index.replace('.jpg', '.png'))
            mask = np.array(Image.open(label_path))
            mask = np.expand_dims(mask, axis=2)
            mask[mask == 255] = self.depth
            class_weights = np.copy(mask)
            class_weights[class_weights < 22] = 1
            class_weights[class_weights == 22] = 0
            # mask = tf.one_hot(mask, self.depth)
            y.append(mask)
            sample_weights.append(class_weights)

        return np.array(y, dtype="float32"), np.array(sample_weights, dtype="float32")







