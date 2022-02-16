import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import ReLU, UpSampling2D, Concatenate, Input, Softmax
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model


def conv_block(
        input,
        fiters=256,
        kernel_size=3,
        padding='same',
        dilation_rate=1,
        use_bias=False
):
    x = Conv2D(
        filters=fiters,
        kernel_size=kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=HeNormal()
    )(input)
    x = BatchNormalization()(x)
    output = ReLU()(x)
    return output


def aspp(input):
    """
    Dilated(Atrous) Spatial Pyramid Pooling
    :return:
    """
    shape = input.shape

    # Image polling
    x_pool = AveragePooling2D((shape[1], shape[2]))(input)
    x_pool = conv_block(x_pool, kernel_size=1)
    x_pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(x_pool)

    # Simple 1X1 convolution
    x_1 = conv_block(input, kernel_size=1, dilation_rate=1)

    # Dilated convolutions with rates: 6, 12, 18
    x_6 = conv_block(input, dilation_rate=6)
    x_12 = conv_block(input, dilation_rate=12)
    x_18 = conv_block(input, dilation_rate=18)

    x = Concatenate()([x_pool, x_1, x_6, x_12, x_18])
    output = conv_block(x, kernel_size=1)

    return output


def deeplabv3plus(image_shape, num_classes):
    inputs = Input(shape=image_shape)

    # Pre-trained ResNet50
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    # Get image features
    x = resnet50.get_layer('conv4_block6_2_relu').output
    x = aspp(x)
    a = UpSampling2D(
        (image_shape[0] // 4 // x.shape[1], image_shape[1] // 4 // x.shape[1]),
        interpolation='bilinear'
    )(x)
    # Get low level image features
    b = resnet50.get_layer('conv2_block3_2_relu').output
    b = conv_block(b, fiters=48, kernel_size=1)
    x = Concatenate()([a, b])
    x = conv_block(x)
    x = conv_block(x)
    x = UpSampling2D(
        (image_shape[0] // x.shape[1], image_shape[1] // x.shape[2]),
        interpolation='bilinear'
    )(x)

    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    # outputs = Softmax()(x)
    outputs = x
    model = Model(inputs, outputs)

    return model


