'''
自定义相关损失函数
'''
from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
    import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_regr_loss(num_anchors):
    '''
    计算RPN网络回归的损失
    :param num_anchors:
    :return:
    '''
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        '''
        对应实际实现的计算RPN网络回归的损失方法
        :param y_true:
            即为之前构造的rpn回归层的标签Y值,shape=(batch_size,height,width, num_anchors*4*2)
            对于y_true来说，最后一个通道的，前4 * num_anchors为是否是正例样本的标记，
            后4 * num_anchors 为实际样本对应真实值。所有最后一个通道个数总共为num_anchors*4*2
        :param y_pred:
            即为样本X经过basenet-rpn网络回归层后的输出值，shape=(batch_size,height,width, num_anchors*4)
        :return:
        '''
        # 这块主要，对于y_true来说，最后一个通道的，前4 * num_anchors为是否是正例正样本的标记，
        # 后4 * num_anchors 为实际样本对应真实值。
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])


    return rpn_loss_regr_fixed_num


def rpn_cls_loss(num_anchors):
    '''
    计算RPN网络分类的损失
    :param num_anchors:
    :return:
    '''
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        # y_true最后一维是2*num_anchors，其中前num_anchors个用来标记对应anchor是否为丢弃不进行训练的anchor
        # 后num_anchors个数据才是真正表示对应anchor是正样本还是负样本
        return lambda_rpn_class * K.sum(
            y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],y_true[:, :, :,num_anchors:])) \
               / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    return rpn_loss_cls_fixed_num


def final_regr_loss(num_classes):
    '''
        计算整个网络最后的回归层对应的损失， 具体计算方法，同rpn中的一样
        :param num_anchors: featuremap 对应每个位置的anchor个数
        :return:
        '''
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def final_cls_loss(y_true, y_pred):
    '''
        计算整个网络最后的分类层对应的损失，直接使用softmax对应的多分类损失函数
        :param y_true:
        :param y_pred:
        :return:
    '''
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
