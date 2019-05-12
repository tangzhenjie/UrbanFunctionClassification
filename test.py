from data.DataGenerator import ImageDataGenerator
from core import model

import tensorflow as tf
import math
slim = tf.contrib.slim

result_txt_file = "D:\\pycharm_program\\UrbanFunctionClassification\\result.txt"
DATASET_DIR = "D:\\competition\\data\\test\\test\\"
CHECKPOINT_DIR = 'D:\\pycharm_program\\UrbanFunctionClassification\\checkpoint\\'
NUM_CLASSES = 9
BATCHSIZE = 1


##################### get the input pipline ############################
with tf.name_scope("input"):
    DataGenerator = ImageDataGenerator(DATASET_DIR, mode="testing")
    # get the dataset statistics
    test_set_length = DataGenerator.test_set_length
    print("test_set_length:%d" % test_set_length)


    TestDataset = DataGenerator.getBatchData(batch_size=BATCHSIZE)

    iterator = TestDataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 100, 100, 3))

with slim.arg_scope(model.vgg_arg_scope()):
    net_output, end_points = model.vgg_16(x, NUM_CLASSES)

# 评价操作
with tf.name_scope("test"):
    pre = tf.argmax(net_output, 1)


##################### setup the network ################################

with tf.Session() as sess:
    # initial variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 判断有没有checkpoint
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored .....")

    # 训练过程
    print("testing start")
    test_batches_of_epoch = int(math.ceil(test_set_length/BATCHSIZE))
    result = []
    for step in range(test_batches_of_epoch):
        img_batch, AreaID_batch = sess.run(next_batch)
        classIDs = sess.run(pre, feed_dict={x: img_batch})
        classID = (classIDs[0] + 1)
        dict_result = {"AreaID": AreaID_batch[0].decode('UTF-8'), "classID": str(classID)}
        result.append(dict_result)
    with open(result_txt_file, "w") as f:
        for item in result:
            f.write(item["AreaID"] + "\t" + item["classID"] + "\n")






