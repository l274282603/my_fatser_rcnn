'''
voc数据集的相关解析
'''
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pprint


def get_data(input_path):
    '''

    :param input_path:  voc数据目录
    :return:
      image_data:解析后的数据集 list列表
      classes_count：一个字典数据结构，key为对应类别名称，value对应为类别所对应的样本（标注框）个数
      classes_mapping：一个字典数据结构，key为对应类别名称，value为对应类别的一个标识index
    '''
    image_data = []
    classes_count = {}  #一个字典，key为对应类别名称，value对应为类别所对应的样本（标注框）个数
    classes_mapping = {} #一个字典数据结构，key为对应类别名称，value为对应类别的一个标识index

    data_paths = os.path.join(input_path, "VOC2012")
    print(data_paths)

    annota_path = os.path.join(data_paths, "Annotations")  # 数据标注目录
    imgs_path = os.path.join(data_paths, "JPEGImages")  # 图片目录

    imgsets_path_train = os.path.join(data_paths, 'ImageSets', 'Main', 'train.txt')
    imgsets_path_val = os.path.join(data_paths, 'ImageSets', 'Main', 'val.txt')
    imgsets_path_test = os.path.join(data_paths, 'ImageSets', 'Main', 'test.txt')
    train_files = []  # 训练集图片名称集合
    val_files = []  # 验证集图片名称集合
    test_files = []  # 测试集图片名称集合

    with open(imgsets_path_train) as f:
        for line in f:
            # strip() 默认去掉字符串头尾的空格和换行符
            train_files.append(line.strip() + '.jpg')

    with open(imgsets_path_val) as f:
        for line in f:
            val_files.append(line.strip() + '.jpg')

    # test-set not included in pascal VOC 2012
    if os.path.isfile(imgsets_path_test):
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')

    # 获得所有的标注文件路径，保存到annota_path_list列表中
    annota_path_list = [os.path.join(annota_path, s) for s in os.listdir(annota_path)]
    index = 0

    # Tqdm 是一个快速，可扩展的Python进度条，
    # 可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
    annota_path_list = tqdm(annota_path_list)

    for annota_path in annota_path_list:
        exist_flag = False
        index += 1
        annota_path_list.set_description("Processing %s" % annota_path.split(os.sep)[-1])

        # 开始解析对应xml数据标注文件
        et = ET.parse(annota_path)
        element = et.getroot()
        element_objs = element.findall("object")  # 获取所有的object子元素
        element_filename = element.find("filename").text  # 对应图片名称
        element_width = int(element.find("size").find("width").text)  # 对应图片尺寸
        element_height = int(element.find("size").find("height").text)  # 对应图片尺寸

        if (len(element_objs) > 0):
            annotation_data = {"filepath": os.path.join(imgs_path, element_filename),
                               "width": element_width,
                               "height": element_height,
                               "image_id": index,
                               "bboxes": []}  # bboxes 用来存放对应标注框的相关位置
        if element_filename in train_files:
            annotation_data["imageset"] = "train"
            exist_flag = True
        if element_filename in val_files:
            annotation_data["imageset"] = "val"
            exist_flag = True
        if len(test_files) > 0:
            if element_filename in test_files:
                annotation_data["imageset"] = "test"
                exist_flag = True

        if not exist_flag:
            continue

        for element_obj in element_objs:  # 遍历一个xml标注文件中的所有标注框
            classes_name = element_obj.find("name").text  # 获取当前标注框的类别名称
            if classes_name in classes_count:  # classes_count 存储类别以及对应类别的标注框个数
                classes_count[classes_name] += 1
            else:
                classes_count[classes_name] = 1

            if classes_name not in classes_mapping:
                classes_mapping[classes_name] = len(classes_mapping)

            obj_bbox = element_obj.find("bndbox")
            x1 = int(round(float(obj_bbox.find("xmin").text)))
            y1 = int(round(float(obj_bbox.find("ymin").text)))
            x2 = int(round(float(obj_bbox.find("xmax").text)))
            y2 = int(round(float(obj_bbox.find("ymax").text)))

            difficulty = int(element_obj.find("difficult").text) == 1
            annotation_data["bboxes"].append({"class": classes_name,
                                              "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                                              "difficult": difficulty})
        image_data.append(annotation_data)

    return image_data, classes_count, classes_mapping


if __name__ == '__main__':
    image_data, classes_count, classes_mapping  = get_data("..\data")
    print("数据集大小：", len(image_data))
    print("类别个数：", len(classes_count))
    print("类别种类：", classes_count.keys())
    print("打印其中一条样本数据：")
    pprint.pprint(image_data[0])
