import collections
import os

from tensorflow.keras import models, layers, backend

from .weights import load_model_weights

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions']
)


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------


def conv_block(filters, stage, block, strides=(2, 2), **kwargs):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), strides=strides, **group_conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)

        x = layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = layers.Conv2D(filters * 2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = layers.BatchNormalization(name=sc_name + '_bn', **bn_params)(shortcut)
        x = layers.Add()([x, shortcut])

        x = layers.Activation('relu', name=relu_name)(x)
        return x

    return layer


def identity_block(filters, stage, block, **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), **group_conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)

        x = layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = layers.Add()([x, input_tensor])

        x = layers.Activation('relu', name=relu_name)(x)
        return x

    return layer


def ResNeXt(
        model_params,
        include_top=True,
        input_shape=None,
        classes=1000,
        weights='imagenet',
        **kwargs):
    img_input = layers.Input(shape=input_shape, name='data')

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnext bottom
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnext body
    init_filters = 128
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1), **kwargs)(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2), **kwargs)(x)

            else:
                x = identity_block(filters, stage, block, **kwargs)(x)

    # resnext top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    # Create model
    model = models.Model(img_input, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top)

    return model


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'resnext50': ModelParams('resnext50', (3, 4, 6, 3)),
    'resnext101': ModelParams('resnext101', (3, 4, 23, 3)),
}


def ResNeXt50(input_shape=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext50'],
        input_shape=input_shape,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNeXt101(input_shape=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext101'],
        input_shape=input_shape,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid'):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).

    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.

    """
    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = layers.Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = layers.Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = layers.Concatenate(axis=slice_axis)(blocks)
        return x

    return layer
