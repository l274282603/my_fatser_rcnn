import os
import cv2
import numpy as np
import sys
import pickle
import time
from keras_faster_rcnn import config, roi_helper, net_model
from keras import backend as K
from keras.layers import Input
from keras.models import Model

config_output_filename = "config/config.pickle"
with open(config_output_filename, "rb") as config_f:
    C = pickle.load(config_f)
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

test_img_path = "test"

class_mapping = C.class_mapping

if "bg" not in class_mapping:
    class_mapping["bg"] = len(class_mapping)

class_mapping = {v:k for k,v in class_mapping.item()}  #key与value调换位置
#class_to_color 定义对应类别多对应的颜色
class_to_color  = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


#定义相关输入Input
img_input = Input(shape=(None, None, 3))
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=(None, None, 512))

#基础网络（VGG）进行特征提取
shared_layers = net_model.base_net_vgg(img_input)

#RPN网络
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layer_out = net_model.rpn_net(shared_layers, num_anchors)

#roi pooling层以及最后网络的输出
final_classifer_reg = net_model.roi_classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layer_out)

model_final_classifer_reg_only = Model([feature_map_input, roi_input], final_classifer_reg)
model_final_classifer_reg = Model([feature_map_input, roi_input], final_classifer_reg)

#加载训练好的模型对应的参数
print("Loading weights from {}".format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_final_classifer_reg.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer="sgd", loss="mse")
model_final_classifer_reg.compile(optimizer="sgd", loss="mse")

all_imgs = []
classes = {}
bbox_threshold = 0.8
visualise = True


def image_Preprocessing(img, C):
    '''
    图片预处理
    :param img:
    :param C:
    :return:
    '''
    height, width, _ = img.shape
    if width < height:
        ratio = float(C.im_size) / width
        new_width = C.im_size
        new_height = int(height * ratio)
    else:
        ratio = float(C.im_size) / height
        new_height = C.im_size
        new_width = int(width * ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    x_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= C.img_channel_mean[0]
    x_img[:, :, 1] -= C.img_channel_mean[1]
    x_img[:, :, 2] -= C.img_channel_mean[2]
    x_img /= C.img_scaling_factor
    x_img = np.expand_dims(x_img, axis=0)
    return x_img, ratio


for idx, img_name in enumerate(sorted(os.listdir(test_img_path))):  #遍历所有测试文件
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print("test image name:{}".format(img_name))
    st = time.time()
    filepath = os.path.join(test_img_path, img_name)

    img = cv2.imread(filepath)  #读取对应图片

    #对测试图片先进行和训练图片一样的预处理
    X, ratio = image_Preprocessing(img, C)

    #经过基础卷积模块和RPN网络后的结果
    [Y1, Y2, feature_map] = model_rpn.predict(X)

    #根据RPN网络结果，获得对应所需要的ROI
    Rois = roi_helper.rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)

    #(x1,y1,x2,y2) to (x,y,w,h)
    Rois[:, 2] -= Rois[:, 0]
    Rois[:, 3] -= Rois[:, 1]

    bboxes = {}
    probs = {}

    for jk in range(Rois.shape[0] // C.num_rois +1):  #一次处理300个roi
        if jk == Rois.shape[0] // C.num_rois:
            rois = np.expand_dims(Rois[C.num_rois * jk:, :], axis=0)
            rois_zero = np.zeros(rois.shape[0], C.num_rois, rois.shape[2])
            rois_zero[:, :rois.shape[1], :] = rois
            rois_zero[:, rois.shape[1]:, :] = rois[0, 0, :]
        else:
            rois = np.expand_dims(Rois[C.num_rois * jk: C.num_rois * (jk + 1), :], axis=0)

        if rois.shape[1] == 0:
            break

        #获得预测结果
        [P_cls, P_regr] = model_final_classifer_reg_only.predict([feature_map, rois])

        for ii in range(P_cls.shape[1]):  #遍历每一个roi对应的预测类别
            #过滤调那些分类概率值不高 以及 负样本
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2]-1):
                continue

            #获得当前roi预测出的类别
            cls_num =np.argmax(P_cls[0,ii, :])
            cls_name = class_mapping[cls_num]
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = rois[0, ii, :]
            tx, ty, tw, th = P_regr[0, ii, 4*cls_num: 4*(cls_num+1)]
            tx /= C.classifier_regr_std[0]
            ty /= C.classifier_regr_std[1]
            tw /= C.classifier_regr_std[2]
            th /= C.classifier_regr_std[3]
            x, y, w, h = roi_helper.apply_regr(x, y, w, h, tx, ty, tw, th)

            #获得预测出来的对应在原始图片上的anchor
            bbox_for_img = [C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)]
            bboxes[cls_name].append(bbox_for_img)
            probs[cls_name].append(cls_num)

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])
        #非极大值抑制
        new_boxes, new_probs = roi_helper.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (class_to_color[key][0], class_to_color[key][1], class_to_color[key][2]), 2)

            textLabel = "{}:{}".format(key, int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))

            retval, baseLine = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('./results_imgs/{}.png'.format(idx), img)













