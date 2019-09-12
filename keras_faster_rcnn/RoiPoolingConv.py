'''
自定义ROI池化层
'''

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np

class RoiPoolingConv(Layer):
    '''
    自定义ROIPooling层
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.dim_ordering = "tf"

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channles = input_shape[0][3]


    def compute_output_shape(self, input_shape):
        '''
        在compute_output_shape方法中实现ROIPooling层的输出
        :param input_shape:
        :return:
        '''
        # 输出5个维度，分别为：[一个batch中的样本个数（图片个数），一个样本对应roi个数，
        #                      每个roi高度，每个roi宽度,通道数]
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channles

    def call(self, x, mask=None):
        '''
        在call方法中实现ROIpooling层的具体逻辑
        :param x:
        :param mask:
        :return:
        '''
        # x 即为传入的模型的输入
        assert(len(x) == 2)

        feature_map = x[0]  #feature map
        rois = x[1]  #输入的所有roi shape=(batchsize, None, 4),最后一维4，代表着对应roi在feature map中的四个坐标值（左上点坐标和宽高）

        input_shape = K.shape(feature_map)
        roi_out_put = []
        for roi_index in range(self.num_rois):
            # print("roi_index=={}".format(roi_index))
            x = rois[0, roi_index, 0]
            y = rois[0, roi_index, 1]
            w = rois[0, roi_index, 2]
            h = rois[0, roi_index, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')
            one_roi_out = tf.image.resize_images(feature_map[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            roi_out_put.append(one_roi_out)

        roi_out_put = tf.reshape(roi_out_put, (self.num_rois, self.pool_size, self.pool_size, self.nb_channles))
        roi_out_put = tf.expand_dims(roi_out_put, axis=0)
        return roi_out_put

    # def call(self, x, mask=None):
    #     '''
    #
    #     :param inputs: 一个list列表，存储着featuremap 和 rois两个张量，具体结构如下：
    #                    [(batch_size, height, width, channel),(batch_size, num_rois, 4)]
    #     :return:  返回 (batch_size, num_rois，pooled_height，pooled_width，n_channels)的张量
    #     '''
    #     assert (len(x) == 2)
    #
    #     def my_pool_rois(x):
    #         return RoiPoolingConv._pool_rois(x[0], x[1], self.pool_size, self.nb_channles)
    #
    #     # 使用「tf.map_fn」生成形状为（batch_size,num_rois，pooled_height，pooled_width，n_channels）的张量。
    #     roi_layer_output = tf.map_fn(fn=my_pool_rois, elems=x, dtype=tf.float32)
    #     return roi_layer_output
    #
    # @staticmethod
    # def _pool_rois( feature_map, rois, pool_size, nb_channles):
    #     '''
    #      #获取单张图像所有RoI 的池化结果
    #     :param feature_map:   一张图片所对应的特征图，shape=(height, width, channel)
    #     :param rois:   一张图片对应的所有roi, shape=(num_rois, 4)
    #     :return:  返回 (num_rois，pooled_height，pooled_width，n_channels)的张量
    #     '''
    #     def my_pool_roi(x):
    #         return RoiPoolingConv._pool_roi(feature_map, x, pool_size, nb_channles)
    #
    #     #使用「tf.map_fn」生成形状为（num_rois，pooled_height，pooled_width，n_channels）的张量。
    #     return tf.map_fn(fn=my_pool_roi, elems=rois, dtype=tf.float32)
    #
    # @staticmethod
    # def _pool_roi(feature_map, roi, pool_size, nb_channles):
    #     '''
    #     #获取单张图像某个RoI 的池化结果
    #     :param feature_map:  一张图片所对应的特征图，shape=(height, width, channel)
    #     :param roi:    某个roi
    #     :return: 返回 (pooled_height，pooled_width，n_channels)的张量
    #     '''
    #     # x,y,w,h即为当前roi 在feature map上的左上点坐标和宽高
    #     x = roi[0]
    #     y = roi[1]
    #     w = roi[2]
    #     h = roi[3]
    #
    #     x = K.cast(x, 'int32')
    #     y = K.cast(y, 'int32')
    #     w = K.cast(w, 'int32')
    #     h = K.cast(h, 'int32')
    #
    #     # 将输入的不同ROI（候选框所对应的featuremap）划分为H * W（pool_size*pool_size）个块
    #     # row_length即为每个块的宽度，col_length为每个块的高度
    #     row_length = K.cast(w / pool_size, 'int32')
    #     col_length = K.cast(h / pool_size, 'int32')
    #
    #     one_roi_out = []
    #     # roi_block_pool = tf.image.resize_images(feature_map[y:y+h, x:x+w, :], (pool_size, pool_size))
    #     # return roi_block_pool
    #
    #
    #     for jy in range(pool_size):
    #         for jx in range(pool_size):
    #             x1 = x + jx * row_length
    #             #这块主要，对于不能完全均等分的，需要判断下靠近右边和下边的块，不能越界或者没包含完整
    #             if (jx + 1) == pool_size:
    #                 x2 = x + w
    #             else:
    #                 x2 = x1 + row_length
    #
    #             y1 = y + jy * col_length
    #             # 这块主要，对于不能完全均等分的，需要判断下靠近右边和下边的块，不能越界或者没包含完整
    #             if (jy + 1) == pool_size:
    #                 y2 = y + h
    #             else:
    #                 y2 = y1 + col_length
    #
    #             x1 = K.cast(x1, 'int32')
    #             x2 = K.cast(x2, 'int32')
    #             y1 = K.cast(y1, 'int32')
    #             y2 = K.cast(y2, 'int32')
    #
    #             roi_block = feature_map[y1:y2, x1:x2, :]
    #             roi_block_pool = K.max(roi_block, axis=(0, 1))
    #             one_roi_out.append(roi_block_pool)
    #     one_roi_out = tf.reshape(one_roi_out, (pool_size, pool_size, nb_channles))
    #     return one_roi_out


if __name__ == '__main__':
    batch_size = 1
    img_height = 200
    img_width = 100
    n_channels = 1
    n_rois = 2
    pooled_size = 7
    feature_maps_shape = (batch_size, img_height, img_width, n_channels)
    feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
    feature_maps_np = np.ones(feature_maps_tf.shape, dtype='float32')
    print(f"feature_maps_np.shape = {feature_maps_np.shape}")
    roiss_tf = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    roiss_np = np.asarray([[[50, 40, 30, 90], [0, 0, 100, 200]]],
                          dtype='float32')
    print(f"roiss_np.shape = {roiss_np.shape}")
    # 创建ROI Pooling层
    roi_layer = RoiPoolingConv(pooled_size, 2)
    pooled_features = roi_layer([feature_maps_tf, roiss_tf])
    print(f"output shape of layer call = {pooled_features.shape}")
    # Run tensorflow session
    with tf.Session() as session:
        result = session.run(pooled_features,
                             feed_dict={feature_maps_tf: feature_maps_np,
                                        roiss_tf: roiss_np})

    print(f"result.shape = {result.shape}")





