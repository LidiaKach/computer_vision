import os
import numpy as np
import shutil
from src.config import IMAGES_DIR, MASKS_DIR, TXT_NAMES_DIR, ROOT_DIR


def train_test_split(img_dir, mask_dir, txt_dir, root_dir, create_dirs=True):
    """
    Function splits folder of images and masks into training/validation sets
    (creates two new folders and copy images and masks into each)

    :param img_dir: string
                    Path to the folder with images
    :param mask_dir: string
                     Path to the folder with masks
    :param txt_dir: string
    :param root_dir: string
                     Path to the root folder in which train/val/test folders will be created
    :param create_dirs: Bool
                        If True train/val/test folders will be created
    """

    if create_dirs:
        # Creating train/val folders
        os.makedirs(root_dir + '/train/images')
        os.makedirs(root_dir + '/train/masks')
        os.makedirs(root_dir + '/val/images')
        os.makedirs(root_dir + '/val/masks')

    # Read the filenames of images and labels into all_names list
    file = open(txt_dir, 'r')
    lines = file.readlines()
    all_names = []
    for line in lines:
        all_names.append(line.replace('\n', ''))
    file.close()

    # Split filenames in train/val (50/50)
    train_names, val_names = np.split(
        np.array(all_names),
        [int(len(all_names) * 0.5)]
    )

    # Copy images and masks into train folder
    for name in train_names:
        shutil.copy(os.path.join(img_dir, name + '.jpg'), os.path.join(root_dir, 'train/images'))
        shutil.copy(os.path.join(mask_dir, name + '.png'), os.path.join(root_dir, 'train/masks'))

    # Copy images and masks into val folder
    for name in val_names:
        shutil.copy(os.path.join(img_dir, name + '.jpg'), os.path.join(root_dir, 'val/images'))
        shutil.copy(os.path.join(mask_dir, name + '.png'), os.path.join(root_dir, 'val/masks'))


if __name__ == '__main__':
    train_test_split(IMAGES_DIR, MASKS_DIR, TXT_NAMES_DIR, ROOT_DIR)
