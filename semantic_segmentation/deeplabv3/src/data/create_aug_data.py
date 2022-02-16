import os
from PIL import Image
import numpy as np
import shutil

from src.data.data_augmentation import aug_img_mask
from src.config import IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, MASKS_AUG_TRAIN_DIR, \
                       IMAGES_AUG_TRAIN_DIR, IMAGES_AUG_VAL_DIR, MASKS_AUG_VAL_DIR



def create_aug_data(
        img_dir, mask_dir, img_train_dir, mask_train_dir,
        img_val_dir, mask_val_dir, create_dirs=True):
    if create_dirs:
        # Creating  folders
        os.makedirs(mask_train_dir)
        os.makedirs(img_train_dir)
        os.makedirs(img_val_dir)
        os.makedirs(mask_val_dir)

    masks_names = os.listdir(mask_dir)

    for mask_name in masks_names:
        name = mask_name.split('.')[0]
        image_name = name + '.jpg'

        mask = np.array(Image.open(mask_dir + mask_name))
        image = Image.open(img_dir + image_name)

        image, mask = aug_img_mask(image, mask, i=0, augment=False)
        img = Image.fromarray(image.astype(np.uint8)).convert('RGB')

        img.save(f'{img_train_dir}{name}.jpg')
        msk = Image.fromarray(mask)
        msk.save(f'{mask_train_dir}{name}.png')

        for i in range(5):
            mask = np.array(Image.open(mask_dir + mask_name))
            image = Image.open(img_dir + image_name)
            image, mask = aug_img_mask(image, mask, i, augment=True)
            img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            img.save(f'{img_train_dir}{name}_{i}.jpg')
            msk = Image.fromarray(mask)
            msk.save(f'{mask_train_dir}{name}_{i}.png')

    masks_names = os.listdir(mask_train_dir)
    print(masks_names)
    np.random.shuffle(masks_names)
    print(masks_names)
    val_names, _ = np.split(
        np.array(masks_names),
        [int(len(masks_names) * 0.1)]
    )
    print(val_names)

    for mask_name in val_names:
        name = mask_name.split('.')[0]
        image_name = name + '.jpg'

        shutil.move(os.path.join(img_train_dir, image_name), img_val_dir)
        shutil.move(os.path.join(mask_train_dir, mask_name), mask_val_dir)




if __name__ == '__main__':
    create_aug_data(
        IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, IMAGES_AUG_TRAIN_DIR,
        MASKS_AUG_TRAIN_DIR, IMAGES_AUG_VAL_DIR, MASKS_AUG_VAL_DIR)
