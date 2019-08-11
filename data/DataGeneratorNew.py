# 这一节我们使用tfrecord来实现读取数据
import os
import math
import cv2
import sys
import numpy as np
import tensorflow as tf
from scipy import io


train_image_dir = "../OriginDataset/train/"
test_image_dir = "../OriginDataset/final_test_images/test/"
txt_dir = "../Dataset/"
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)



#########生成feature方法##########
class DataGenerator(object):
    def __init__(self, ratio=19/20):

        # 根据比例分成训练集和测试集，（并把所有测试集）生成目录文件（测试集还没写）
        train_txt_filename = "train.txt"
        eval_txt_filename = "eval.txt"
        test_txt_filename = "test.txt"
        self.train_txt_path = txt_dir + train_txt_filename
        self.eval_txt_path = txt_dir + eval_txt_filename
        self.test_txt_path = txt_dir + test_txt_filename
        self._get_txt_file(ratio)

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
                    if key >= tag:
                        eval_File.write("train_part/" + file_name[-9] + "/" + file_name[: -4] + ".txt" + "\n")
                    else:
                        train_File.write("train_part/" + file_name[-9] + "/" + file_name[: -4] + ".txt" + "\n")
            train_File.close()
            eval_File.close()
        if not os.path.exists(self.test_txt_path):
            #下面可以写生成test数据集的txt代码，后期再写
            test_subdirs = os.listdir(test_image_dir)
            with open(self.test_txt_path, "w") as f:
                for file_name in test_subdirs:
                    f.write(file_name[:-4] + ".txt" + "\n")
if __name__ == "__main__":
    DataGenerator = DataGenerator()