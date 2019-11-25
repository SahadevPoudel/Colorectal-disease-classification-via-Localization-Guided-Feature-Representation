from keras.layers import Input,Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D,AtrousConvolution2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras_drop_block import DropBlock2D


def convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X
    if stage == 4 or stage == 5:
        dilation = 2
    else:
        dilation = 1
    ##### MAIN PATH #####
    # First component of main path
    X = AtrousConvolution2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = AtrousConvolution2D(filters=F2, kernel_size=(f, f), atrous_rate=(dilation, dilation), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = AtrousConvolution2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = AtrousConvolution2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    if stage == 4 or stage == 5:
        X_shortcut = DropBlock2D(block_size=7, keep_prob=0.9)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    if stage == 4:
        dilation = 2

    elif stage == 5:
        if  block == 'b' or block == 'c':
            dilation = 4
        elif block == 'd':
            dilation = 2
        else:
            dilation = 1
    else:
        dilation = 1
    # First component of main path
    X = AtrousConvolution2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = AtrousConvolution2D(filters=F2, kernel_size=(f, f), atrous_rate=(dilation, dilation), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = AtrousConvolution2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    if stage == 4 or stage == 5:
        X = DropBlock2D(block_size=7, keep_prob=0.9)(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    if stage == 5:
        if block == 'e':
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)
        else:
            X = Activation('relu')(X)
    return X

def ResNet50(input_shape=(224, 224, 3), classes=5):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(X)

    # # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=1)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    #
    # # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=1)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='d')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='e')

    X = GlobalAveragePooling2D()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model