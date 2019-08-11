import tensorflow as tf
import os
import random
import sys
import pandas as pd
import cv2
import numpy as np


def get_data(dataset):
    print("Loading training set...")
    table = pd.read_csv(dataset, header=None)
    filenames = ["../OriginDataset/train/" + item[0][-7: -4] + "/" + item[0][-14:-4] + ".jpg" for item in table.values]                                                        # 存储方式为  ‘/train_image/008/032316_008.jpg’,‘’,‘’,''
    class_ids = [int(item[0][-5])-1 for item in table.values]     # 存储功能分类的id,从0开始到8 [0,0,0,...., 1,1,1...,2,2,2...,3,3,3...,4,4,4..,...8,8,8],每个id的个数为9342个
    data = []
    for index, filename in enumerate(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)                           # 读入一副彩色图像。图像的透明度会被忽略，(为什么透明度要被忽略，，可不可以换成)
        # image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)                     # 原始图片读入
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)                       # 以灰色图片读入
        visit = np.load("../Dataset/npy/train_visit/"+filename.split('/')[-1].split('.')[0]+".npy")    # 032316_008.npy  里面的数据是一个三位矩阵
        label = class_ids[index]
        data.append([image, visit.astype(np.int32), label])
    random.seed(0)
    random.shuffle(data)
    print("Loading completed...")
    return data                                                                     # 格式是一个数组[[imgae, visit, label],[,,,],...]

def get_data_test(dataset):
    print("Loading testing set...")
    table = pd.read_csv(dataset, header=None)     # 读取train_oversampling.txt的数据
    filenames = ["../OriginDataset/test/" + item[0].split('/')[-1].split('.')[0] + ".jpg" for item in table.values]                                                        # 存储方式为  ‘/train_image/008/032316_008.jpg’,‘’,‘’,''
    class_ids = [int(item[0].split('/')[-1].split('.')[0]) for item in table.values]     # 存储功能分类的id,从0开始到8 [0,0,0,...., 1,1,1...,2,2,2...,3,3,3...,4,4,4..,...8,8,8],每个id的个数为9342个
    data = []
    for index, filename in enumerate(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)                           # 读入一副彩色图像。图像的透明度会被忽略，(为什么透明度要被忽略，，可不可以换成)
        # image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)                     # 原始图片读入
        # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)                       # 以灰色图片读入
        visit = np.load("../Dataset/npy/test_visit/" + filename.split('/')[-1].split('.')[0] +".npy")    # 032316_008.npy  里面的数据是一个三位矩阵
        areaID = class_ids[index]
        data.append([image, visit.astype(np.int32), areaID])
    print("Loading completed...")
    return data

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(data, visit, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(data),
        'visit': bytes_feature(visit),
        'label': int64_feature(label),
    }))



def _convert_dataset(data, tfrecord_path, dataset):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, dataset+".tfrecord")    # /home/wangdong/桌面/工程项目目录Spyter-tensorblow/研究生竞赛 /一带一路竞赛/初赛赛题/tfrecord/train.tfrecord
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)        # 创建一个writer来写TFRecords文件
    length = len(data)                                                    # 三维数组的长度   84078
    for index, item in enumerate(data):
        data_ = item[0].tobytes()
        visit = item[1].tobytes()
        label = item[2]                                                   # 对应功能分类的标签
        example = image_to_tfexample(data_, visit, label)
        tfrecord_writer.write(example.SerializeToString())                # 将样列序列化为字符串后， 写入out_filename文件中
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

def _convert_dataset_test(data, tfrecord_path, dataset):
    """ Convert data to TFRecord format. """
    output_filename = os.path.join(tfrecord_path, dataset+".tfrecord")    # /home/wangdong/桌面/工程项目目录Spyter-tensorblow/研究生竞赛 /一带一路竞赛/初赛赛题/tfrecord/train.tfrecord
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)        # 创建一个writer来写TFRecords文件
    length = len(data)                                                    # 三维数组的长度   84078
    for index, item in enumerate(data):
        data_ = item[0].tobytes()
        visit = item[1].tobytes()
        areaID = item[2]                                              # 对应地区的名字
        example = image_to_tfexample(data_, visit, areaID)
        tfrecord_writer.write(example.SerializeToString())                # 将样列序列化为字符串后， 写入out_filename文件中
        sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    if not os.path.exists("../Dataset/tfrecord/"):
        os.makedirs("../Dataset/tfrecord/")

    data = get_data("../Dataset/train.txt")
    _convert_dataset(data, "../Dataset/tfrecord/", "train")

    data = get_data("../Dataset/eval.txt")
    _convert_dataset(data, "../Dataset/tfrecord/", "eval")

    data = get_data_test("../Dataset/test.txt")
    _convert_dataset_test(data, "../Dataset/tfrecord/", "test")

