import numpy as np
import pdb
import math
# from . import data_generators
import copy
from keras_faster_rcnn import data_generators



def apply_regr_np(X, T):
    '''
    通过rpn网络的回归层的预测值，来调整anchor位置
    :param X:
    :param T:
    :return:
    '''
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[:, :, 0]
        ty = T[:, :, 1]
        tw = T[:, :, 2]
        th = T[:, :, 3]

        # (cx, cy)原始anchor中心点位置
        cx = x + w/2.
        cy = y + h/2.

        #(cx1, cy1)经过rpn网络回归层调整后，anchor中心点位置
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w  #经过rpn网络回归层调整后，anchor 宽度
        h1 = np.exp(th.astype(np.float64)) * h  #经过rpn网络回归层调整后，anchor 高度
        #（x1，y1）经过rpn网络回归层调整后，anchor的左上点坐标
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    '''
    非极大值抑制算法，提取出300个anchor作为输入roipooling层的roi
    简单介绍下非极大值抑制算法，假如当前有10个anchor，根据是正样本的概率值进行升序排序为[A,B,C,D,E,F,G,H,I,J]
    1.从具有最大概率的anchor J开始，计算其余anchor与J之间的iou值
    2.如果iou值大于overlap_thresh阈值，则删除掉，并将当前J重新保留下来，使我们需要的。
      例如，如果D,F与J之间的iou大于阈值，则直接舍弃，同时把J重新保留，也从原始数组中删除掉。
    3.在剩余的[A,B,C,E,G,H]中，继续选取最大的概率值对应的anchor,然后重复上述过程。
    4.最后，当数组为空，或者保留下来的anchor个数达到设定的max_boxes，则停止迭代，
      最终保留下的来的anchor 就是最终需要的。

    :param boxes: #经过rpn网络后生成的所有候选框,shape = (anchor个数，4)
    :param probs: #rpn网络分类层的输出值，value对应是正例样本的概率，shape = (anchor个数，)
    :param overlap_thresh:  iou阈值
    :param max_boxes:  最大提取的roi个数
    :return:
    '''
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    boxes = boxes.astype("float")

    pick = []

    area = (x2 - x1) * (y2 - y1)  #所有anchor的各自的区域面积（anchor个数，）

    #将所有anchor根据概率值进行升序排序
    idxs = np.argsort(probs)  #默认是升序

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]  #最后一个索引，即为当前idxs中具体最大概率值（是否为正例）的anchor的索引
        pick.append(i)  #保留当前anchor对应索引

        # 计算当前选取出来的anchor与其他anchor之间的交集
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])
        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int  #当前选取出来的索引对应的anchor,与其他anchor之间的 交集

        # 计算当前选取出来的索引对应的anchor 与其他anchor之间的并集
        area_union = area[i] + area[idxs[:last]] - area_int

        #overlap 即为当前选取出来的索引对应的anchor 与其他anchor之间的交并比（iou）
        overlap = area_int/(area_union + 1e-6)

        #在idxs中删除掉与当前选取出来的anchor之间iou大于overlap_thresh阈值的。
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:  #如果当前保留的anchor个数已经达到max_boxes，则直接跳出迭代
            break


    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

def rpn_to_roi(rpn_cls_layer, rpn_regr_layer, C, use_regr=True, max_boxes=300,overlap_thresh=0.9):
    '''
    建立rpn网络与roi pooling层的连接
    通过rpn网络的输出，找出对应的roi
    :param rpn_cls_layer:  rpn网络的分类输出
    :param rpn_regr_layer:  rpn网络的回归输出
    :param C:
    :param dim_ordering:
    :param use_regr:
    :param max_boxes:
    :param overlap_thresh:
    :return:
    '''
    regr_layer = rpn_regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_cls_layer.shape[0] == 1
    (rows, cols) = rpn_cls_layer.shape[1:3]

    curr_layer = 0
    # A.shape = (4个在feature_map上的对应位置信息（左上角和右下角坐标）， feature_map_height, feature_map_wigth, k(9))
    A = np.zeros((4, rpn_cls_layer.shape[1], rpn_cls_layer.shape[2], rpn_cls_layer.shape[3]))
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride   #对应anchor在feature map上的宽度
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride   #对应anchor在feature map上的高度
            # if dim_ordering == 'th':
            #     regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            # else:
            #     regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  #当前anchor对应回归值
            #     regr = np.transpose(regr, (2, 0, 1))
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  # 当前anchor对应回归值
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2   #左上点横坐标
            A[1, :, :, curr_layer] = Y - anchor_y/2   #左上纵横坐标
            A[2, :, :, curr_layer] = anchor_x   #暂时存储anchor 宽度
            A[3, :, :, curr_layer] = anchor_y   #暂时存储anchor 高度

            if use_regr:
                #通过rpn网络的回归层的预测值，来调整anchor位置
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]  #右下角横坐标
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]  #右下角纵坐标

            #确保anchor不超过feature map尺寸
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    #将对应shape调整到二维（anchor总共个数，4）
    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_cls_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    #过滤掉一些异常的框
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    #通过非极大值抑制，选取出一些anchor作为roipooling层的输入
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result


def calc_roi(R, img_data, C, class_mapping):
    '''
    生成roipooing层的输入数据以及最终分类层的训练数据Y值以及最终回归层的训练数据Y值
    :param R:  通过rpn网络输出结果，选取出来的对应rois,shape=(rois个数，4)
    :param img_data:  经过相关预处理后的原始数据，格式如下：
    {'width': 500,
      'height': 500,
      'bboxes': [{'y2': 500, 'y1': 27, 'x2': 183, 'x1': 20, 'class': 'person', 'difficult': False},
                 {'y2': 500, 'y1': 2, 'x2': 249, 'x1': 112, 'class': 'person', 'difficult': False},
                 {'y2': 490, 'y1': 233, 'x2': 376, 'x1': 246, 'class': 'person', 'difficult': False},
                 {'y2': 468, 'y1': 319, 'x2': 356, 'x1': 231, 'class': 'chair', 'difficult': False},
                 {'y2': 450, 'y1': 314, 'x2': 58, 'x1': 1, 'class': 'chair', 'difficult': True}], 'imageset': 'test',
      'filepath': './datasets/VOC2007/JPEGImages/000910.jpg'
    }
    :param C: 存储相关配置信息
    :param class_mapping: 一个字典数据结构，key为对应类别名称，value为对应类别的一个标识
    :return:
    '''

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])

    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    #获得真实标注框在feature map上的坐标
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []

    for ix in range(R.shape[0]):  #遍历所有Roi
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0  #用来存储当前roi(候选框)与所有真实标注框之间的最优iou值
        best_bbox = -1  #当前roi(候选框)对应的最优候选框index
        for bbox_num in range(len(bboxes)):  #遍历所有真实标注框
            #计算真实标注框与roi（候选框）之间的iou值
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:

                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                # （tx, ty, tw, th）即为此roi到ground-truth（真实检测框）的对应4个平移缩放参数
                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label)) # y_class_num即为构造的最终分类层的训练数据Y值
        coords = [0] * 4 * (len(class_mapping) - 1)  # 每个类别4个坐标值
        labels = [0] * 4 * (len(class_mapping) - 1)  # 对应存储类别标签值
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)  #roipooling层输入
    Y1 = np.array(y_class_num)  #最终分类层的训练样本Y值
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)   #最终回归层的训练样本Y值

    # np.expand_dims 统一增加一维，minibatch
    X = np.expand_dims(X, axis=0)
    Y1 = np.expand_dims(Y1, axis=0)
    Y2 = np.expand_dims(Y2, axis=0)

    # neg_samples: 负样本在第二维的所有index列表
    # pos_samples: 正样本在第二维的所有index列表
    neg_samples = np.where(Y1[0, :, -1] == 1)  # 最后一个数值为1，说明是负样本
    pos_samples = np.where(Y1[0, :, -1] == 0)
    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []
    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    # len(pos_samples) ：负样本个数
    # len(pos_samples)： 正样本个数
    if len(pos_samples) < C.num_rois // 2:  # 如果正样本个数少于150，则所有正样本都参与训练
        selected_pos_samples = pos_samples.tolist()
    else:  # 否则的话，随机抽取150个正样本
        selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
    try:
        # replace=False 无放回抽取
        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                replace=False).tolist()
    except:
        #  replace=True 有放回抽取
        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                replace=True).tolist()

    # sel_samples： 参与训练的roi样本对应的下标
    sel_samples = selected_pos_samples + selected_neg_samples


    return X[:, sel_samples, :], Y1[:, sel_samples, :], Y2[:, sel_samples, :], IoUs


if __name__ == '__main__':
    pass
    # a = np.array([[[1,2,3],[2,3,4],[3,4,5]],[[0,1,2],[2,1,0],[0,2,0]]])
    # a = np.array([[[0, 1, 0], [0, 0, 1], [0, 0, 1]]])
    # print(a)
    # print(a.shape)
    # print(a[0,:,-1])
    # result = np.where(a[0,:,-1] == 1)
    # print(result)