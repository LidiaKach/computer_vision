IMAGE_SHAPE = (512, 512, 3)
IMG_SIZE = 512
BATCH_SIZE = 32
NUM_CLASSES = 21

ROOT_DIR = '../../data_voc/'

IMAGES_DIR = '../../data_voc/JPEGImages/'
MASKS_DIR = '../../data_voc/SegmentationClass/'

TXT_NAMES_DIR = '../../data_voc/ImageSets/Segmentation/trainval.txt'

IMAGES_TRAIN_DIR = '../../data_voc/train/images/'
MASKS_TRAIN_DIR = '../../data_voc/train/masks/'

IMAGES_AUG_TRAIN_DIR = '../../data_voc/train_aug/images/'
MASKS_AUG_TRAIN_DIR = '../../data_voc/train_aug/masks/'
IMAGES_AUG_VAL_DIR = '../../data_voc/val_aug/images/'
MASKS_AUG_VAL_DIR = '../../data_voc/val_aug/masks/'

IMAGES_VAL_DIR = '../../data_voc/val/images/'
MASKS_VAL_DIR = '../../data_voc/val/masks/'
