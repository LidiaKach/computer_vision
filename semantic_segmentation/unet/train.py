import tensorflow as tf
import albumentations as A

from model import unet
from datagenerator import CarvanaDataset
from utils import plot_history
from config import TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, IMG_HEIGHT, IMG_WIDTH, IMG_SHAPE

if __name__ == '__main__':
    train_transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
    ])

    model = unet(IMG_SHAPE)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  sample_weight_mode='temporal',
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    train = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    valid = CarvanaDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=train_transform)

    # fit the generators
    history = model.fit(
        train,
        validation_data=valid,
        epochs=20,
        callbacks=[callback]
    )

    # Save trained model
    model.save('model.h5')

    plot_history(history)






