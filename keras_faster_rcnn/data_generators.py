from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
import threading
import itertools
from keras_faster_rcnn import data_augment


def get_new_img_size(width, height, img_min_size=600):
    if width <= height:
        f = float(width / img_min_size)
        resized_height = int(f * height)
        resized_width = int(img_min_size)
    else:
        f = float(height / img_min_size)
        resized_width = int(f * width)
        resized_height = int(img_min_size)
    return resized_width, resized_height

#计算两个框之前的并集
def union(au, bu, area_intersection):
    # au和bu的格式为： (x1,y1,x2,y2)
    # area_intersection为 au 和 bu 两个框的交集
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

#计算两个框之前的交集
def intersection(ai, bi):
    # ai和bi的格式为： (x1,y1,x2,y2)
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


# 计算两个框的iou值
def iou(a, b):
    # a和b的格式为： (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)  #计算交集
    area_u = union(a, b, area_i)  #计算并集

    return float(area_i) / float(area_u + 1e-6)   #交并比


def getdata_for_rpn(config, img_data, width, heigth, resized_width, resized_height):
    '''
        用于提取RPN网络训练集,也就是产生各种anchors以及anchors对应与ground truth的修正参数
        :param C:   配置信息
        :param img_data:  原始数据
        :param width:    缩放前图片的宽
        :param heigth:   缩放前图片的高
        :param resized_width:  缩放后图片的宽
        :param resized_height:  缩放后图片的高
        :param img_length_calc_function:  获取经过base Net后提取出来的featur map图像尺寸，
                                          对于VGG16来说，就是在原始图像尺寸上除以16
        :return:
    '''
    downscale = float(config.rpn_stride)   #原始图像到feature map之间的缩放映射关系
    anchor_sizes = config.anchor_box_scales   #anchor 三种尺寸
    anchor_ratios = config.anchor_box_ratios  # anchor 三种宽高比
    num_anchors = len(anchor_sizes) * len(anchor_ratios)  # 每一个滑动窗口所对应的anchor个数，也就是论文中的k值

    #计算出经过base Net后提取出来的featurmap图像尺寸
    output_width = int(resized_width / 16)
    output_height = int(resized_height / 16)

    # （36,36,9），用来存放RPN网络，训练样本最后分类层输出时的y值，
    # 最后一维9代表对于每个像素点对应9个anchor,值为0或1（正样本或负样本）
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))

    #(36,36,9），用来存放对于每个anchor,是否是有效的样本，值为0或者1（无效样本，有效样本）
    # 因为对于iou在0.3到0.7之间的样本，是直接丢弃 ，不参与训练的
    # 另外，只是在一张图片中随机采样256个anchor,其他的也不参与训练
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))

    # （36,36,9*4），用来存放RPN网络，针对一张图片，最后回归层的标签Y值
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    #获取一张训练图片的真实标注框个数，也就是含有的待检测的目标个数
    num_bboxes = len(img_data['bboxes'])

    # 用来存储每个bbox（真实标注框）所对应的anchor个数
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)

    # 用来存储每个bbox（真实标注框）所对应的最优anchor在feature map中的位置信息，以及大小信息
    # [jy, ix, anchor_ratio_idx, anchor_size_idx]
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)

    # 每个bbox（真实标注框）与所有anchor 的最优IOU值
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)

    # 用来存储每个bbox（真实标注框）所对应的最优anchor的坐标值
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)

    # 用来存储每个bbox（真实标注框）与所对应的最优anchor之间的4个平移缩放参数，用于回归预测
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    gta = np.zeros((num_bboxes, 4))  # 用来存放经过缩放后的标注框
    # 因为之前图片进行了缩放，所以需要将对应的标注框做对应调整
    for bbox_num, bbox in enumerate(img_data["bboxes"]):
        gta[bbox_num, 0] = bbox["x1"] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox["x2"] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox["y1"] * (resized_height / float(heigth))
        gta[bbox_num, 3] = bbox["y2"] * (resized_height / float(heigth))


    #遍历feature map上的每一个像素点
    for ix in range(output_width):
        for iy in range(output_height):
            #在feature map的每一个像素点上，遍历对应不同大小，不同长宽比的k(9)个anchor
            for anchor_size_index in range(len(anchor_sizes)):
                for anchor_ratio_index in range(len(anchor_ratios)):
                    anchor_x = anchor_sizes[anchor_size_index] * anchor_ratios[anchor_ratio_index][0]
                    anchor_y = anchor_sizes[anchor_size_index] * anchor_ratios[anchor_ratio_index][1]

                    # 获得当前anchor在原图上的X坐标位置
                    # downscale * (ix + 0.5)即为当前anchor在原始图片上的中心点X坐标
                    # downscale * (ix + 0.5) - anchor_x / 2 即为当前anchor左上点X坐标
                    # downscale * (ix + 0.5) + anchor_x / 2 即为当前anchor右下点X坐标
                    x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                    x2_anc = downscale * (ix + 0.5) + anchor_x / 2
                    # 去掉那些跨过图像边界的框
                    if x1_anc<0 or x2_anc > resized_width:
                        continue

                    # 获得当前anchor在原图上的Y坐标位置
                    # downscale * (jy + 0.5)即为当前anchor在原始图片上的中心点Y坐标
                    # downscale * (jy + 0.5) - anchor_y / 2 即为当前anchor左上点Y坐标
                    # downscale * (jy + 0.5) + anchor_y / 2 即为当前anchor右下点Y坐标
                    y1_anc = downscale * (iy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (iy + 0.5) + anchor_y / 2
                    # 去掉那些跨过图像边界的框
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # 用来存放当前anchor类别是前景(正样本)还是背景（负样本）
                    bbox_type = "neg"
                    # best_iou_for_loc 是用来存储当前anchor针对于所有真实标注框的一个最优iou
                    # 需要与前面的best_iou_for_bbox 每个真实标注框 针对于所有 anchor 的最优iou是不一样的。
                    best_iou_for_loc = 0.0

                    #遍历所有真实标注框，也就是所有ground truth
                    for bbox_num in range(num_bboxes):
                        # 计算当前anchor与当前真实标注框的iou值
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])

                        #根据iou值，判断当前anchor是否为正样本。
                        # 如果是，则计算此anchor(正样本)到ground - truth（真实检测框）的对应4个平移缩放参数。
                        # 判断一个anchor是否为正样本的两个条件为：
                        # 1.与ground - truth（真实检测框）IOU最高的anchor
                        # 2.与任意ground - truth（真实检测框）的IOU大于0.7 的anchor
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > config.rpn_max_overlap:
                            # cx,cy: ground-truth（真实检测框）的中心点坐标
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            # cxa,cya: 当前anchor的中心点坐标
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            # （tx, ty, tw, th）即为此anchor(正样本)到ground-truth（真实检测框）的对应4个平移缩放参数
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data["bboxes"][bbox_num]["class"] != "bg":
                            #针对于当前ground - truth（真实检测框），如果当前anchor与之的iou最大，则重新更新相关存储的best值
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [iy, ix, anchor_ratio_index, anchor_size_index]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                            #对于iou大于0.7的，则，无论是否是最优的,直接认为是正样本
                            if curr_iou > config.rpn_max_overlap:
                                bbox_type = "pos"
                                num_anchors_for_bbox[bbox_num] +=1
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)#当前anchor与和它有最优iou的那个ground-truth（真实检测框）之间的对应4个平移参数
                            # iou值大于0.3，小于0.7的的，即不是正样本，也不是负样本
                            if config.rpn_min_overlap < curr_iou < config.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'
                    if bbox_type == "neg":
                        test_index = anchor_size_index * len(anchor_ratios) + anchor_ratio_index
                        y_is_box_valid[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 1
                        y_rpn_overlap[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 0
                    elif bbox_type == "neutral":
                        y_is_box_valid[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 0
                        y_rpn_overlap[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 0
                    elif bbox_type == "pos":
                        y_is_box_valid[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 1
                        y_rpn_overlap[iy, ix, anchor_size_index * len(anchor_ratios) + anchor_ratio_index] = 1
                        start = 4 * (anchor_size_index * len(anchor_ratios) + anchor_ratio_index)
                        y_rpn_regr[iy, ix, start:start+4] = best_regr

    # 经过上面，我们只是挑选出了 与任意ground - truth（真实检测框）的IOU大于0.7 的anchor为正样本。
    # 但是如果某个ground - truth（真实检测框） 没有与它iou值大于0.7的anchor呢？
    # 这个时候需要用到第一个条件 与ground - truth（真实检测框）IOU最高的anchor
    # 我们需要确保每一个真实标注框都有至少一个对应的正样本（anchor）
    for idx in range(num_anchors_for_bbox.shape[0]):#遍历所有真实标注框，也就是所有ground truth
        if num_anchors_for_bbox[idx] == 0:  #如果当前真实标注框没有所对应的anchor
            if best_anchor_for_bbox[idx, 0] == -1: #如果当前真实标注框没有与任何anchor都无交集，也就是说iou都等于0，则直接忽略掉
                continue
            y_is_box_valid[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                           best_anchor_for_bbox[idx, 3] * len(anchor_ratios) + best_anchor_for_bbox[idx,2]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                           best_anchor_for_bbox[idx, 3] * len(anchor_ratios) + best_anchor_for_bbox[idx, 2]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 3] * len(anchor_ratios) + best_anchor_for_bbox[idx, 2])
            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start+4] \
                = best_dx_for_bbox[idx, :]

    #增加一维，（样本个数）
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    '''
    a = np.array([[0,1,0],
                 [1,0,1]])
    b = np.array([[1,1,0],
                 [0,1,1]])
    print(np.logical_and(a, b))
    # [[False  True False]
    #  [False False  True]]
    print(np.where(np.logical_and(a, b)))  
    #(array([0, 1], dtype=int64), array([1, 2], dtype=int64))
    '''
    #np.asarray(condition).nonzero()
    #pos_locs 正样本对应的三个维度的下标
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    #neg_locs 负样本对应的三个维度的下标
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])  #正样本个数

    # 随机采样256个样本作为一个mini-batch,并且最多保持正负样本比例1:1,如果正样本个数不够，用负样本填充
    mini_batch = 256
    if len(pos_locs[0]) > mini_batch / 2:  #判断正样本个数是否多于128，如果是，则从所有正样本中随机采用128个
        # 注意这块，是从所有正例的下标中留下128个，选取出其他剩余的，将对应的y_is_box_valid设置为0，
        # 也就是说选取出来的正例样本就是丢弃的， 不进行训练的样本，剩余的128个即为实际的正例样本
        val_locs = random.sample(range(num_pos), num_pos - mini_batch / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0

        num_pos = mini_batch / 2

    # 正样本选取完毕后，开始选取负例样本，同样的思路，随机选取出不需要的负样本，将对应的y_is_box_valid设置为0，
    # 剩余的正好和正样本组合成 256个样本
    if len(neg_locs[0]) + num_pos > mini_batch:
        #(mini_batch-num_pos) : 需要的负例样本数
        #len(neg_locs[0]) - (mini_batch-num_pos)：不需要的负例样本数
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - (mini_batch-num_pos))
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    #将y_is_box_valid 与 y_rpn_overlap连接到一块
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3)

    #对于回归损失来说，只是针对正样本进行计算的，负样本和不参与训练的其他样本都需要过滤掉，不参与训练
    #所以这块需要将y_rpn_overlap 和 y_rpn_regr拼接起来作为RPN网络回归层的真实Y值，方便后续计算损失函数
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=3), y_rpn_regr], axis=3)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)



def get_anchor_data_gt(img_datas, class_count, C, mode="train"):
    '''
    生成最终训练数据集的迭代器
    :param img_data:  原始数据，list,每个元素都是一个字典类型，存放着每张图片的相关信息
    all_img_data[0] = {'width': 500,
                       'height': 500,
                       'bboxes': [{'y2': 500, 'y1': 27, 'x2': 183, 'x1': 20, 'class': 'person', 'difficult': False},
                                  {'y2': 500, 'y1': 2, 'x2': 249, 'x1': 112, 'class': 'person', 'difficult': False},
                                  {'y2': 490, 'y1': 233, 'x2': 376, 'x1': 246, 'class': 'person', 'difficult': False},
                                  {'y2': 468, 'y1': 319, 'x2': 356, 'x1': 231, 'class': 'chair', 'difficult': False},
                                  {'y2': 450, 'y1': 314, 'x2': 58, 'x1': 1, 'class': 'chair', 'difficult': True}], 'imageset': 'test',
                        'filepath': './datasets/VOC2007/JPEGImages/000910.jpg'}
    :param class_count:  数据集中各个类别的样本个数，字典型
    :param C:            相关配置参数
    :param mode:
    :return: 返回一个数据迭代器
    '''
    while True:
        if mode == "train":
            #打乱数据集
            random.shuffle(img_datas)

        for img_data in img_datas:
            try:
                pass
                #数据增强
                if mode == "train":
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                #确保图像尺寸不发生改变
                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape
                assert cols == width
                assert rows == height

                #将图像的短边缩放到600尺寸
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_cls, y_rpn_regr = getdata_for_rpn(C, img_data_aug, width, height, resized_width, resized_height)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                yield  np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
            except Exception as e:
                print(e)
                continue
