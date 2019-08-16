from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras import backend as K
from keras_faster_rcnn import RoiPoolingConv


def base_net_vgg(input_tensor):
    if input_tensor is None:
        input_tensor = Input(shape=(None,None,3))
    else:
        if not K.is_keras_tensor(input_tensor):
            input_tensor = Input(tensor=input_tensor, shape=(None,None,3))

    #开始构造基础模型（VGG16的卷积模块）,到block5_conv3层，用来提取feature map

    # Block 1
    X = Conv2D(filters=64, kernel_size=(3,3), activation="relu",
               padding="same", name="block1_conv1")(input_tensor)
    X = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
               padding="same", name="block1_conv2")(X)
    X = MaxPool2D(pool_size=(2,2), strides=(2,2), name="block1_pool")(X)

    # Block 2
    X = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
               padding="same", name="block2_conv1")(input_tensor)
    X = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
               padding="same", name="block2_conv2")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(X)

    # Block 3
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv1")(input_tensor)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv2")(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
               padding="same", name="block3_conv3")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(X)

    # Block 4
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv1")(input_tensor)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block4_conv3")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(X)

    # Block 5
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv1")(input_tensor)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation="relu",
               padding="same", name="block5_conv3")(X)
    return X


def rpn_net(shared_layers, num_anchors):
    '''
    RPN网络
    :param shared_layers: 共享层的输出，作为RPN网络的输入（也就是VGG的卷积模块提取出来的feature map）
    :param num_anchors:  feature map中每个位置所对应的anchor个数（这块为9个）
    :return:
    [X_class, X_regr, shared_layers]：分类层输出（二分类，这块使用sigmoid）,回归层输出，共享层
    '''
    X = Conv2D(512, (3,3), padding="same", activation="relu",
               kernel_initializer="normal", name="rpn_conv1")(shared_layers)
    #采用多任务进行分类和回归
    X_class = Conv2D(num_anchors, (1,1), activation="sigmoid",
                     kernel_initializer="uniform", name="rpn_out_class")(X)
    X_regr = Conv2D(num_anchors*4, (1,1), activation="linear",
                    kernel_initializer="zero",name="rpn_out_regress")(X)
    return [X_class, X_regr, shared_layers]

def roi_classifier(shared_layers, input_rois, num_rois, nb_classes=21):
    '''
    最后的检测网络（包含ROI池化层 和 全连接层）,进行最终的精分类和精回归
    :param shared_layers: 进行特征提取的基础网络（VGG的卷积模块）
    :param input_rois:  roi输入 shape=(None, 4)
    :param num_rois:  roi数量
    :param nb_classes: 总共的待检测类别，需要算上 背景类
    :return:  [out_class, out_regr]：最终分类层输出和回归层输出
    '''
    #ROI pooling层
    pooling_regions = 7
    roi_pool_out = RoiPoolingConv(pooling_regions, num_rois)([shared_layers, input_rois])

    #全连接层
    out = Flatten(name="flatten")(roi_pool_out)
    out = Dense(4096, activation="relu", name="fc1")(out)
    out = Dense(4096, activation="relu", name="fc2")(out)

    out_class = Dense(nb_classes, activation="softmax", name='dense_class_{}'.format(nb_classes))(out)
    out_regr = Dense(4 * (nb_classes-1), activation="linear", name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

