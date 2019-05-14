#coding=utf-8
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework import dtypes

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    """DataGenerator CLass

    """

    def __init__(self, data_dir, mode, ratio=4/5, txt_dir="D:\\pycharm_program\\UrbanFunctionClassification\\data\\"):

        """Recieves a dir to imagedir and a ratio of training set and total set to get a
        train text file and a val text file

        :param data_dir: a dir to imagedir
        :param radio: a ratio of training set and verification set
        """
        self.mode = mode
        if mode == "training":
            train_txt_filename = "train.txt"
            eval_txt_filename = "eval.txt"
            self.train_txt = txt_dir + train_txt_filename
            self.eval_txt = txt_dir + eval_txt_filename
            self.dataset_dir = data_dir
            self._get_txt_file(data_dir, train_txt_filename=self.train_txt, eval_txt_filename=self.eval_txt, ratio=ratio)
            self._read_txt_file()
        elif mode == "testing":
            # 表示是测试集
            test_txt_filename = "test.txt"
            self.test_txt_filename = txt_dir + test_txt_filename
            self.dataset_dir = data_dir
            self._get_txt_file(data_dir, test_txt_filename=self.test_txt_filename)
            self._read_txt_file()




    def _get_txt_file(self, data_dir, test_txt_filename=None, train_txt_filename=None, eval_txt_filename=None, ratio=None):
        mode = self.mode

        if mode == "training":
            if not os.path.exists(train_txt_filename):
                subdirs = os.listdir(data_dir)

                train_File = open(train_txt_filename, "w")
                eval_File = open(eval_txt_filename, "w")
                for subdir in subdirs:
                    # the category in this subdir
                    category = subdir[-1]
                    current_dir = os.path.join(data_dir, subdir)
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
        elif mode == "testing":
            # 测试集
            if not os.path.exists(test_txt_filename):
                current_dir_filenames = os.listdir(data_dir)
                with open(test_txt_filename, "w") as f:
                    for file_name in current_dir_filenames:
                        AreaID = file_name.split(".")[0]
                        f.write(file_name + "\t" + AreaID + "\n")



    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        mode = self.mode
        if mode == "training":
            self.train_img_paths = []
            self.train_labels = []
            self.eval_img_paths = []
            self.eval_labels = []

            with open(self.train_txt, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split("\t")
                    self.train_img_paths.append(os.path.join(self.dataset_dir, items[0]))
                    self.train_labels.append(int(items[1][0]) - 1)
            with open(self.eval_txt, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split("\t")
                    self.eval_img_paths.append(os.path.join(self.dataset_dir, items[0]))
                    self.eval_labels.append(int(items[1][0]) - 1)
            self.train_set_length = len(self.train_img_paths)
            self.eval_set_length = len(self.eval_img_paths)
        elif mode == "testing":
            self.test_img_paths = []
            self.test_areaIDs = []
            with open(self.test_txt_filename, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split("\t")
                    self.test_img_paths.append(os.path.join(self.dataset_dir, items[0]))
                    self.test_areaIDs.append(items[1].split("\n")[0])
            self.test_set_length = len(self.test_img_paths)
    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.train_img_paths
        labels = self.train_labels
        permutation = np.random.permutation(len(self.train_labels))
        self.train_img_paths = []
        self.train_labels = []
        for i in permutation:
            self.train_img_paths.append(path[i])
            self.train_labels.append(labels[i])

    def getBatchData(self, type, batch_size=None, num_classes=9,  shuffle=False, buffer_size=500):

        self.num_classes = num_classes

        # create dataset
        if type == "training":
            # 注意训练时，获取验证集时 shuffle必须是False
            if shuffle:
                self._shuffle_lists()
            # convert lists to TF tensor
            self.train_img_paths = convert_to_tensor(self.train_img_paths, dtype=dtypes.string)
            self.train_labels = convert_to_tensor(self.train_labels, dtype=dtypes.int32)

            train_dataset = tf.data.Dataset.from_tensor_slices((self.train_img_paths, self.train_labels))
            data = train_dataset.map(self._parse_function_train)
            # shuffle the first `buffer_size` elements of the dataset
            if shuffle:
                data = data.shuffle(buffer_size=buffer_size)
            # create a new dataset with batches of images
            data = data.repeat(None).batch(batch_size)

        elif type == 'inference':
            # convert lists to TF tensor
            self.eval_img_paths = convert_to_tensor(self.eval_img_paths, dtype=dtypes.string)
            self.eval_labels = convert_to_tensor(self.eval_labels, dtype=dtypes.int32)

            eval_dataset = tf.data.Dataset.from_tensor_slices((self.eval_img_paths, self.eval_labels))
            data = eval_dataset.map(self._parse_function_inference)
            # shuffle the first `buffer_size` elements of the dataset
            if shuffle:
                data = data.shuffle(buffer_size=buffer_size)
            # create a new dataset with batches of images
            data = data.repeat(1).batch(batch_size)
        elif type == "testing":
            # convert lists to TF tensor
            self.test_img_paths = convert_to_tensor(self.test_img_paths, dtype=dtypes.string)
            self.test_areaIDs = convert_to_tensor(self.test_areaIDs, dtype=dtypes.string)

            test_dataset = tf.data.Dataset.from_tensor_slices((self.test_img_paths, self.test_areaIDs))
            data = test_dataset.map(self._parse_function_test)
            data = data.batch(batch_size)
        else:
            raise ValueError("Invalid mode '%s'." % (type))


        return data

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)

        """
        Dataaugmentation comes here.
        """
        img_decoded = tf.cast(img_decoded, tf.float32)
        img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)


        return img_centered, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)

        img_decoded = tf.cast(img_decoded, tf.float32)
        img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)

        return img_centered, one_hot
    def _parse_function_test(self, filename, areaID):
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)

        img_decoded = tf.cast(img_decoded, tf.float32)
        img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)

        return img_centered, areaID


#if __name__ == "__main__":
#    x = ImageDataGenerator("D:\\competition\\data\\test\\test\\", mode="testing")

