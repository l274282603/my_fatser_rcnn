import cv2
import numpy as np
import random
import pprint
from keras_faster_rcnn import voc_data_parser, data_augment, data_generators, config

'''
 all_img_data = [{'width': 500,
                 'height': 500,
                 'bboxes': [{'y2': 500, 'y1': 27, 'x2': 183, 'x1': 20, 'class': 'person', 'difficult': False},
                            {'y2': 500, 'y1': 2, 'x2': 249, 'x1': 112, 'class': 'person', 'difficult': False},
                            {'y2': 490, 'y1': 233, 'x2': 376, 'x1': 246, 'class': 'person', 'difficult': False},
                            {'y2': 468, 'y1': 319, 'x2': 356, 'x1': 231, 'class': 'chair', 'difficult': False},
                            {'y2': 450, 'y1': 314, 'x2': 58, 'x1': 1, 'class': 'chair', 'difficult': True}], 
                 'imageset': 'test',
                 'filepath': './datasets/VOC2007/JPEGImages/000910.jpg'
                 }
                 ...
                 ]
'''
if __name__ == '__main__':
    image_data, classes_count, classes_mapping = voc_data_parser.get_data("data")
    print("数据集大小：", len(image_data))
    print("类别个数：", len(classes_count))
    print("类别种类：", classes_count.keys())
    print("打印其中一条样本数据：")
    pprint.pprint(image_data[3])
    print("所有类别个数：")
    pprint.pprint(classes_count)
    # img = cv2.imread(image_data[3]["filepath"])
    #
    # bboxes = image_data[3]["bboxes"]
    # for bbox in bboxes:
    #     cv2.rectangle(img, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)
    # cv2.imshow("1", img)
    #
    config = config.Config()
    # img_data_aug, img = data_augment.augment(image_data[3], config)
    #
    # bboxes = img_data_aug["bboxes"]
    # for bbox in bboxes:
    #     cv2.rectangle(img, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)
    # cv2.imshow("2", img)
    # cv2.waitKey()
    # image_data, classes_count, classes_mapping = voc_data_parser.get_data("data")
    train_imgs = [s for s in image_data if s['imageset'] == 'train']   #训练集
    val_imgs = [s for s in image_data if s['imageset'] == 'val']  #验证集
    test_imgs = [s for s in image_data if s['imageset'] == 'test'] #测试集
    data_gen_train = data_generators.get_anchor_data_gt(train_imgs[:3], classes_count, config)
    for i in range(3):
        X, Y, img_data = next(data_gen_train)
        print("经过预处理后的图像X：",X.shape)
        print("RPN网络分类层对应Y值：",Y[0].shape)
        print("RPN网络回归层层对应Y值：",Y[1].shape)