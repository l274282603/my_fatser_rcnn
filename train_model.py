from keras_faster_rcnn import config, data_generators, data_augment, losses
from keras_faster_rcnn import  net_model, roi_helper, RoiPoolingConv, voc_data_parser
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import generic_utils
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import numpy as np
import time
import pprint
import pickle
#获取原始数据集
all_imgs, classes_count, class_mapping = voc_data_parser.get_data("data")
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

pprint.pprint(classes_count)
print('类别数 (包含背景) = {}'.format(len(classes_count)))

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'train']  #训练集
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']  #验证集
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']  #测试集
print('训练样本个数 {}'.format(len(train_imgs)))
print('验证样本个数 {}'.format(len(val_imgs)))
print('测试样本个数 {}'.format(len(test_imgs)))

C = config.Config()  #相关配置信息
C.class_mapping = class_mapping
config_output_filename = "config/config.pickle"
with open(config_output_filename, "wb") as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))


#生成用于RPN网络训练数据集的迭代器
data_gen_train = data_generators.get_anchor_data_gt(train_imgs, classes_count, C, mode='train')
data_gen_val = data_generators.get_anchor_data_gt(val_imgs, classes_count, C, mode='val')
data_gen_test = data_generators.get_anchor_data_gt(test_imgs, classes_count, C, mode='val')

img_input = Input(shape=(None, None, 3))  #网络模型最开始的输入
roi_input = Input(shape=(None, 4))   #roi模块的输入

'''
model_rpn : 输入：图片数据；  输出：对应RPN网络中分类层和回归层的两个输出
model_classifier：  输入： 图片数据和选取出来的ROI数据；   输出： 最终分类层输出和回归层输出
'''
# 用来进行特征提取的基础网络 VGG16
shared_layers = net_model.base_net_vgg(img_input)
# RPN网络
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = net_model.rpn_net(shared_layers, num_anchors)
# 最后的检测网络（包含ROI池化层 和 全连接层）
classifier = net_model.roi_classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

#这是一个同时包含RPN和分类器的模型，用于为模型加载/保存权重
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    print('loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

except:
    print('没有找到上一次的训练模型')
    try:
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('没有找到预训练的模型参数')


optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_cls_loss(num_anchors), losses.rpn_regr_loss(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.final_cls_loss, losses.final_regr_loss(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 1000  #每1000轮训练，记录一次平均loss
num_epochs = 2000
iter_num = 0
train_step = 0  #记录训练次数

losses = np.zeros((epoch_length, 5))  #用来存储1000轮训练中，没一轮的损失
# rpn_accuracy_rpn_monitor = []
# rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

print('Starting training')
for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    while True:
        # if len(rpn_accuracy_rpn_monitor) == epoch_length:
        #     mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
        #     rpn_accuracy_rpn_monitor = []
        #     print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
        #         mean_overlapping_bboxes, epoch_length))
        #     if mean_overlapping_bboxes == 0:
        #         print(
        #             'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

        X, Y, img_data = next(data_gen_train)  #通过构造的迭代器，获得一条数据
        # print(X.shape)
        # print(Y[0].shape, Y[1].shape)
        loss_rpn = model_rpn.train_on_batch(X, Y)  #训练basenet 与 RPN网络

        P_rpn = model_rpn.predict_on_batch(X)  #获得RPN网络的输出

        #通过rpn网络的输出，找出对应的roi
        R = roi_helper.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7,
                                   max_boxes=300)
        #生成roipooing层的输入数据以及最终分类层的训练数据Y值以及最终回归层的训练数据Y值
        X2, Y1, Y2, IouS = roi_helper.calc_roi(R, img_data, C, class_mapping)

        if X2 is None:
            continue
        # print("model_classifier.train_on_batch--X.shape={},X2.shape={}".format(X.shape, X2.shape))
        loss_class = model_classifier.train_on_batch([X, X2], [Y1, Y2])
        train_step += 1

        losses[iter_num, 0] = loss_rpn[1]  #rpn_cls_loss
        losses[iter_num, 1] = loss_rpn[2]  #rpn_regr_loss

        losses[iter_num, 2] = loss_class[1]  #final_cls_loss
        losses[iter_num, 3] = loss_class[2]  #final_regr_loss
        losses[iter_num, 4] = loss_class[3]  #final_acc

        iter_num += 1

        progbar.update(iter_num,
                       [('rpn_cls', np.mean(losses[:iter_num, 0])),
                        ('rpn_regr', np.mean(losses[:iter_num, 1])),
                        ('detector_cls', np.mean(losses[:iter_num, 2])),
                        ('detector_regr', np.mean(losses[:iter_num, 3]))])

        if iter_num == epoch_length:     #每1000轮训练，统计一次
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            # mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            # rpn_accuracy_for_epoch = []

            if C.verbose:
                # print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            iter_num = 0
            start_time = time.time()

            if curr_loss < best_loss:
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                best_loss = curr_loss
                model_all.save_weights(C.model_path)

            break