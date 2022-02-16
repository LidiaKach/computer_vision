import tensorflow as tf
import matplotlib.pyplot as plt

from src.model.deeplabv3plus import deeplabv3plus
from src.config import IMAGE_SHAPE, NUM_CLASSES, MASKS_AUG_TRAIN_DIR, \
                       IMAGES_AUG_TRAIN_DIR, IMAGES_AUG_VAL_DIR, MASKS_AUG_VAL_DIR
from src.data.data_generator import DataGenerator


def plot_history(history):
    # Show plots of accuracy and loss vs. epoch
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

if __name__ == '__main__':
    model = deeplabv3plus(IMAGE_SHAPE, NUM_CLASSES)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  sample_weight_mode='temporal',
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    train = DataGenerator(IMAGES_AUG_TRAIN_DIR, MASKS_AUG_TRAIN_DIR, NUM_CLASSES + 1)
    valid = DataGenerator(IMAGES_AUG_VAL_DIR, MASKS_AUG_VAL_DIR, NUM_CLASSES + 1)

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