import tensorflow as tf


def ref_padding(tensor, pad_size=1):
    output = tf.pad(tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")
    return output


def conv_conv_pool_refPad(input_, n_filters, name, pool=True, activation=tf.nn.leaky_relu):
    """{Conv -> LeakyRELU -> BN}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(ref_padding(net),F, (3, 3),activation=None, padding='valid',name="conv_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat_refPad(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D_refPad(inputA, input_B, n_filter, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D_refPad(tensor, target_tensor, n_filter, name):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    net_shape = tf.shape(target_tensor)
    tensor = tf.image.resize_bilinear(tensor, size=(net_shape[1], net_shape[2]))

    # [N, H, W, C] = target_tensor.shape
    # tensor = upsample2D_tensorflow(tensor)
    # tensor = tf.image.resize_bilinear(tensor, size=(H, W))

    return  tf.layers.conv2d(
                    ref_padding(tensor),
                    n_filter, (3, 3),
                    activation=None,
                    padding='valid',
                    name="upsample_{}".format(name))


def unet(X, name, outputChn, reuse):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # net = X / 127.5 - 1  # image input
        # net = X * 2.0 - 1  # sigmoid input
        conv1, pool1 = conv_conv_pool_refPad(X, [32, 32], name=1)
        conv2, pool2 = conv_conv_pool_refPad(pool1, [64, 64], name=2)
        conv3, pool3 = conv_conv_pool_refPad(pool2, [128, 128], name=3)
        conv4, pool4 = conv_conv_pool_refPad(pool3, [256, 256], name=4)
        conv5 = conv_conv_pool_refPad(
            pool4, [512, 512], name=5, pool=False)

        up6 = upconv_concat_refPad(conv5, conv4, 256, name=6)
        conv6 = conv_conv_pool_refPad(up6, [256, 256], name=6, pool=False)

        up7 = upconv_concat_refPad(conv6, conv3, 128, name=7)
        conv7 = conv_conv_pool_refPad(up7, [128, 128], name=7, pool=False)

        up8 = upconv_concat_refPad(conv7, conv2, 64, name=8)
        conv8 = conv_conv_pool_refPad(up8, [64, 64], name=8, pool=False)

        up9 = upconv_concat_refPad(conv8, conv1, 32, name=9)
        conv9 = conv_conv_pool_refPad(up9, [32, 32], name=9, pool=False)
        result = tf.layers.conv2d(
            conv9,
            outputChn, (1, 1),
            name='final',
            activation=None,
            padding='same')

        return result