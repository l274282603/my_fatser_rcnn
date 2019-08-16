'''
自定义相关损失函数
'''
import tensorflow as tf
import numpy as np
from keras import backend as K

lambda_rpn_regr = 10.0
epsilon = 1e-4

def rpn_regr_loss(num_anchors):
    '''
    计算RPN网络回归的损失
    :param num_anchors: featuremap 对应每个位置的anchor个数
    :return:
    '''
    def rpn_regr_loss_fixed_num(y_true, y_pred):
        '''
        实际实现的计算RPN网络回归的损失方法
        :param y_true:
            即为之前构造的rpn回归层的标签Y值,shape=(batch_size,height,width, num_anchors*4*2)
            对于y_true来说，最后一个通道的，前4 * num_anchors为是否是正例样本的标记，
            后4 * num_anchors 为实际样本对应真实值。所有最后一个通道个数总共为num_anchors*4*2
        :param y_pred:
            即为样本X经过basenet-rpn网络回归层后的输出值，shape=(batch_size,height,width, num_anchors*4)
        :return:
        '''
        x = y_true[:, :, :, 4*num_anchors:] - y_pred
        x_abs = tf.abs(x)
        x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32)
        loss = lambda_rpn_regr * tf.reduce_sum(
            y_true[:, :, :, :4*num_anchors] * (x_bool * (0.5*x*x) + (1-x_bool)*(x_abs-0.5))) / \
               tf.reduce_sum(epsilon + y_true[:, :, :, :4*num_anchors])
        return loss
    return rpn_regr_loss_fixed_num


def rpn_cls_loss(num_anchors):
    '''
    计算RPN网络分类的损失
    :param num_anchors: featuremap 对应每个位置的anchor个数
    :return:
    '''
    def rpn_cls_loss_fixed_num(y_true, y_pred):
        '''
        实际实现的计算RPN网络分类的损失方法
        :param y_true:
                  即为之前构造的rpn分类层的标签Y值,shape=(batch_size, height, width, num_anchors*2)
                  其中前num_anchors个用来标记对应anchor是否为丢弃不进行训练的anchor
                  后num_anchors个数据才是真正表示对应anchor是正样本还是负样本
        :param y_pred:
                  即为样本X经过basenet-rpn网络分类层后的输出值，shape=(batch_size, height, width, num_anchors)
        :return:
        '''
        loss = tf.reduce_sum(
            y_true[:, :, :, :num_anchors] * tf.nn.sigmoid_cross_entropy_with_logits(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) \
               / tf.reduce_sum(epsilon + y_true[:, :, :, :num_anchors])
        return loss


def final_regr_loss(num_anchors):
    '''
    计算整个网络最后的回归层对应的损失， 具体计算方法，同rpn中的一样
    :param num_anchors: featuremap 对应每个位置的anchor个数
    :return:
    '''
    def final_regr_loss_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_anchors:] - y_pred
        x_abs = tf.abs(x)
        x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32)
        loss = lambda_rpn_regr * tf.reduce_sum(
            y_true[:, :, :4*num_anchors] * (x_bool * (0.5*x*x) + (1-x_bool)*(x_abs-0.5))) / \
               tf.reduce_sum(epsilon + y_true[:, :, :4*num_anchors])
        return loss
    return final_regr_loss_fixed_num

def class_loss_cls(y_true, y_pred):
    '''
    计算整个网络最后的分类层对应的损失，直接使用softmax对应的多分类损失函数
    :param y_true:
    :param y_pred:
    :return:
    '''
    return K.mean(K.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))