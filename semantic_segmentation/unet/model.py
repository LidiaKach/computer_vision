import tensorflow as tf


def conv_block(filters):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


def upsample(filters):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=2,
                                        padding='same', use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def unet(shape, features=[64, 128, 256, 512]):
    inputs = tf.keras.layers.Input(shape=shape)
    x = inputs

    down_stack = []
    for feature in features:
        down_stack.append(conv_block(feature))

    up_stack = []
    for feature in reversed(features):
        up_stack.append(upsample(feature))

    up_conv = []
    for feature in reversed(features):
        up_conv.append(conv_block(feature))

    bottleneck = conv_block(features[-1] * 2)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D()(x)

    x = bottleneck(x)
    skips = skips[::-1]

    # Upsampling and establishing the skip connections
    for up, skip, conv in zip(up_stack, skips, up_conv):
        x = up(x)
        if x.shape != skip.shape:
            x = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(x)

        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv(x)
    outputs = tf.keras.layers.Conv2D(1, (1, 1))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test():
    x = tf.random.normal(3, 165, 210, 3)
    model = unet([165, 210, 3])
    pred = model(x)
    assert x.shape == pred.shape


if __name__ == '__main':
    test()
