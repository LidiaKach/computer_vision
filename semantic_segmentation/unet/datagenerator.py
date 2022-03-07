import os
import tensorflow as tf
from PIL import Image
import numpy as np


class CarvanaDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, fit=True, transform=None, batch_size=32):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.transform = transform
        self.fit = fit
        self.images = os.listdir(image_dir)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        X = []
        y = []
        idxs = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.fit:
            for index in idxs:
                img_path = os.path.join(self.image_dir, index)
                mask_path = os.path.join(self.mask_dir, index.replace('.jpg', '_mask.gif'))
                image = np.array(Image.open(img_path).convert('RGB'))
                mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
                mask[mask == 255.0] = 1.0
                if self.transform:
                    augmentations = self.transform(image=image, mask=mask)
                    image = augmentations['image']
                    mask = augmentations['mask']
                X.append(image)
                y.append(mask)
            return np.array(X, dtype="float32"), np.array(y, dtype="float32")
        else:
            for index in idxs:
                img_path = os.path.join(self.image_dir, index)
                image = np.array(Image.open(img_path).convert('RGB'))
                X.append(image)
            return np.array(X, dtype="float32")








