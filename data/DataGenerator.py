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

    def __init__(self, data_dir, ratio=4/5, txt_dir="D:\\pycharm_program\\UrbanFunctionClassification\\data\\"):

        """Recieves a dir to imagedir and a ratio of training set and total set to get a
        train text file and a val text file

        :param data_dir: a dir to imagedir
        :param radio: a ratio of training set and verification set
        """
        train_txt_filename = "train.txt"
        eval_txt_filename = "eval.txt"
        self.train_txt = txt_dir + train_txt_filename
        self.eval_txt = txt_dir + eval_txt_filename
        self.dataset_dir = data_dir
        self._get_txt_file(data_dir, self.train_txt, self.eval_txt, ratio)
        self._read_txt_file()

    def _get_txt_file(self, data_dir, train_txt_filename, eval_txt_filename, ratio):
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


    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
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

    def getBatchData(self, batch_size, num_classes, mode="training", shuffle=True, buffer_size=500):

        self.num_classes = num_classes

        # create dataset
        if mode == 'training':
            self._shuffle_lists()
            # convert lists to TF tensor
            self.train_img_paths = convert_to_tensor(self.train_img_paths, dtype=dtypes.string)
            self.train_labels = convert_to_tensor(self.train_labels, dtype=dtypes.int32)

            train_dataset = tf.data.Dataset.from_tensor_slices((self.train_img_paths, self.train_labels))
            data = train_dataset.map(self._parse_function_train)

        elif mode == 'inference':
            # convert lists to TF tensor
            self.eval_img_paths = convert_to_tensor(self.eval_img_paths, dtype=dtypes.string)
            self.eval_labels = convert_to_tensor(self.eval_labels, dtype=dtypes.int32)

            eval_dataset = tf.data.Dataset.from_tensor_slices((self.eval_img_paths, self.eval_labels))
            data = eval_dataset.map(self._parse_function_inference)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        # create a new dataset with batches of images
        data = data.repeat(None).batch(batch_size)
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



