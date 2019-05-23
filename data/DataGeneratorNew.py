# 这一节我们使用tfrecord来实现读取数据
import os
import math
import cv2
import sys
import numpy as np
import tensorflow as tf
from data import VisitGenerator
from scipy import io


train_image_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\train_img\\train\\"
train_visit_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\train_visit\\train\\"
test_image_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\test_img\\test\\"
test_visit_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\test_visit\\test\\"
txt_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\"
tfrecord_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\"
mat_dir = "D:\\pycharm_program\\UrbanFunctionClassification\\Dataset\\"
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
NUM_CLASSES = 9             # 数据集类别数用于生成one_hot数据
#########生成feature方法##########
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _datas_to_tfexample(data, visit, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': _bytes_feature(data),
        'visit': _bytes_feature(visit),
        'label': _int64_feature(label),
    }))
def _tf_record_parser(record):
    keys_to_features = {
        'data': tf.FixedLenFeature([], tf.string),
        'visit': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(features['data'], tf.uint8)    # 保存时是uint8所以这必须是uint8(位数必须一样否则报错)
    image = tf.reshape(image, [100, 100, 3])
    image = tf.cast(image, tf.float32)
    img_centered = tf.subtract(image, IMAGENET_MEAN)
    # 在这里可以对图像进行处理（现在我们暂且不处理）
    visit = tf.decode_raw(features['visit'], tf.int64)   # 保存时是int64所以这必须是64位
    visit = tf.reshape(visit, [182, 24])
    visit = tf.cast(visit, tf.float32)
    # 可以在这里改变visit特征的形状使用tf.reshape()
    label = tf.cast(features['label'], tf.int64)
    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return img_centered, visit, label_onehot

#########生成feature方法##########
class DataGenerator(object):
    def __init__(self, ratio=4/5):

        # 根据比例分成训练集和测试集，（并把所有测试集）生成目录文件（测试集还没写）
        train_txt_filename = "train.txt"
        eval_txt_filename = "eval.txt"
        self.train_txt_path = txt_dir + train_txt_filename
        self.eval_txt_path = txt_dir + eval_txt_filename
        self._get_txt_file(ratio)

        # 根据训练集和验证集目录文件来生成对应的mat（测试集还没有写）
        train_mat_filename = "train.mat"
        eval_mat_filename = "eval.mat"
        self.train_mat_path = mat_dir + train_mat_filename
        self.eval_mat_path = mat_dir + eval_mat_filename
        self._get_mat_file()
        train_tfrecord_filename = "train.tfrecord"
        eval_tfrecord_filename = "eval.tfrecord"
        self.train_tfrecord_path = tfrecord_dir + train_tfrecord_filename
        self.eval_tfrecord_path = tfrecord_dir + eval_tfrecord_filename
        self._get_tfrecord_file()
    def _get_txt_file(self, ratio):
        """
        根据比例生成目录文件（ratio = traindataset_length / total）
        :param ratio:
        :return:
        """
        # 先判断该文件是否存在
        if not os.path.exists(self.train_txt_path):
            subdirs = os.listdir(train_image_dir)

            train_File = open(self.train_txt_path, "w")
            eval_File = open(self.eval_txt_path, "w")
            for subdir in subdirs:
                # the category in this subdir
                category = subdir[-1]
                current_dir = os.path.join(train_image_dir, subdir)
                current_dir_filenames = os.listdir(current_dir)

                tag = math.ceil((len(current_dir_filenames) * ratio))
                for key, file_name in enumerate(current_dir_filenames):
                    current_file_path = os.path.join(subdir, file_name)
                    if key >= tag:
                        eval_File.write(current_file_path + "\t" + category + "\n")
                    else:
                        train_File.write(current_file_path + "\t" + category + "\n")
            train_File.close()
            eval_File.close()
            #下面可以写生成test数据集的txt代码，后期再写
    def _get_mat_file(self):
        if not os.path.exists(self.train_mat_path):
            """如果不存在我们就生成"""

            # 首先生成训练集的mat!!!!
            # 注意mat文件的格式:
            #               array1{“文件名”：[182,24]}
            #               array2{“文件名”：[182*24]})
            #               array12{“文件名”:[[182,24],[182*24]]}

            # 我们现在只提取第一种特征（注意现在mat文件中的键值是目录文件的名字，并排序是一样的）
            VisitGenerator.GetStatisticDataArray1FromVisit(self.train_txt_path, train_visit_dir, self.train_mat_path)
            VisitGenerator.GetStatisticDataArray1FromVisit(self.eval_txt_path, train_visit_dir, self.eval_mat_path)
    def _get_tfrecord_file(self):
        # 生成train eval test tfrecord文件
        # 分三步 第一步生成feature 第二步生成tf.example 第三步：保存文件
        # 获取要保存的数据
        if not os.path.exists(self.train_tfrecord_path):
            train_record_data = self._get_tfrecord_data(tag="training")
            eval_record_dataeval = self._get_tfrecord_data(tag="evaling")
            self._save_data_to_tfrecord_file(train_record_data, tag="training")
            self._save_data_to_tfrecord_file(eval_record_dataeval, tag="evaling")

    def _get_tfrecord_data(self, tag="training"):
        result = []
        if tag == "training":
            visits = io.loadmat(self.train_mat_path)
            with open(self.train_txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split("\t")
                    key = items[0].split("\\")[1].split(".")[0]   # 不同系统上\\不同
                    image_path = os.path.join(train_image_dir, items[0])
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    label = int(items[1][0]) - 1
                    visit = visits[key]
                    result.append([image, visit, label])

            permutation = np.random.permutation(len(result))
            result_shuffled = []
            for i in permutation:
                result_shuffled.append(result[i])

        elif tag == "evaling":
            visits = io.loadmat(self.eval_mat_path)
            with open(self.eval_txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split("\t")
                    key = items[0].split("\\")[1].split(".")[0]  # 不同系统上\\不同
                    image_path = os.path.join(train_image_dir, items[0])
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    label = int(items[1][0]) - 1
                    visit = visits[key]
                    result.append([image, visit, label])
            result_shuffled = result
        # 测试集后期写
        return result_shuffled
    def _save_data_to_tfrecord_file(self, data, tag="training"):
        output_filename = ""
        if tag == "training":
            output_filename = self.train_tfrecord_path
        elif tag == "evaling":
            output_filename = self.eval_tfrecord_path
        #  生成tfrecord文件
        tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
        length = len(data)
        for index, item in enumerate(data):
            image_data = item[0].tobytes()
            visit = item[1].tobytes()
            label = item[2]
            # 生成example
            example =_datas_to_tfexample(image_data, visit, label)
            tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
            sys.stdout.flush()


    def get_batch(self, batch_size, tag="training"):
        if tag == "training":
            training_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            training_dataset = training_dataset.map(_tf_record_parser)
            training_dataset = training_dataset.repeat(None)
            training_dataset = training_dataset.shuffle(buffer_size=500)
            training_dataset = training_dataset.batch(batch_size)

            return training_dataset
        if tag == "evaling":
            evaling_dataset = tf.data.TFRecordDataset(self.eval_tfrecord_path)
            evaling_dataset = evaling_dataset.map(_tf_record_parser)
            evaling_dataset = evaling_dataset.repeat(1)
            evaling_dataset = evaling_dataset.shuffle(buffer_size=500)
            evaling_dataset = evaling_dataset.batch(batch_size)
            return evaling_dataset
#DataGenerator()
#train_lenth = len([x for x in tf.python_io.tf_record_iterator(tfrecord_dir + "train.tfrecord")])
#eval_lenth = len([x for x in tf.python_io.tf_record_iterator(tfrecord_dir + "eval.tfrecord")])
#i = 0
#DataGenerator = DataGenerator()
#dataset = DataGenerator.get_batch(10, tag="training")
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#with tf.Session() as sess:
#    image, visit, label = sess.run(next_element)
 #   i == value
#VisitGenerator.GetStatisticDataArray1FromVisit(txt_dir + "eval.txt", train_visit_dir, "zhenjie.mat")