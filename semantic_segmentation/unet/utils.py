import os
import numpy as np
import shutil
import matplotlib.pyplot as plt



def train_test_split(root_dir, create_dirs=True):
    """
    Function splits folder of images and masks into training/validation/test sets
    (creates two new folders and move images and masks into each)

    :param root_dir: string
                     Path to the root folder in which val/test folders will be created
    :param create_dirs: Bool
                        If True val/test folders will be created
    """

    train_dir_images = os.path.join(root_dir, 'train/images')
    train_dir_masks = os.path.join(root_dir, 'train/masks')
    val_dir_images = os.path.join(root_dir, 'val/images')
    val_dir_masks = os.path.join(root_dir, 'val/masks')
    test_dir_images = os.path.join(root_dir, 'test/images')
    test_dir_masks = os.path.join(root_dir, 'test/masks')


    if create_dirs:
        # Creating test/val folders
        os.makedirs(test_dir_images)
        os.makedirs(test_dir_masks)
        os.makedirs(val_dir_images)
        os.makedirs(val_dir_masks)


    names = os.listdir(train_dir_images)

    # Split filenames in train/val/test (80/10/10)
    train_names, val_names, test_names = np.split(
        np.array(names),
        [int(len(names) * 0.8), int(len(names) * 0.9)]
    )

    # Move images and masks into test folder
    for name in test_names:
        shutil.move(os.path.join(train_dir_images, name),  test_dir_images)
        shutil.move(os.path.join(train_dir_masks, name.replace('.jpg', '_mask.gif')), test_dir_masks)

    # Move images and masks into val folder
    for name in val_names:
        shutil.move(os.path.join(train_dir_images, name), val_dir_images)
        shutil.move(os.path.join(train_dir_masks, name.replace('.jpg', '_mask.gif')), val_dir_masks)


def plot_history(history):
    """Show plots of accuracy and loss vs. epoch"""
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(121)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'valid'], loc='upper left')
    ax2 = fig.add_subplot(122)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'valid'], loc='upper left')
    plt.show()


