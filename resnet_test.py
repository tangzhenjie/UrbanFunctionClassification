from data import DataGeneratorNew
from core import ResNet

import tensorflow as tf
import math
slim = tf.contrib.slim

result_txt_file = "result.txt"
DATASET_DIR = "D:\\competition\\data\\test_img\\test"
CHECKPOINT_DIR = 'D:\\pycharm_program\\UrbanFunctionClassification\\checkpoint'
NUM_CLASSES = 9
BATCHSIZE = 1
def tool(AreaID):
    last = 6 - len(AreaID)
    for key in range(last):
        AreaID  = "0" + AreaID
    return AreaID

##################### get the input pipline ############################
with tf.name_scope("input"):
    DataGenerator = DataGeneratorNew.DataGenerator()
    TestDataset = DataGenerator.get_batch(BATCHSIZE, tag="testing")
    # get the dataset statistics
    test_set_length = 10000
    print("test_set_length:%d" % test_set_length)

    iterator = TestDataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 100, 100, 3))
is_training = tf.placeholder('bool', [])

with tf.name_scope("ResNet"):
    depth = 50    # 可以是50、101、152
    ResNetModel = ResNet.ResNetModel(is_training, depth, NUM_CLASSES)
    net_output = ResNetModel.inference(x)

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
        img_batch, visit, AreaID_batch = sess.run(next_batch)
        classIDs = sess.run(pre, feed_dict={x: img_batch, is_training: False})
        classID = (classIDs[0] + 1)
        print("AreaID %d" % AreaID_batch[0])
        print(classID)
        dict_result = {"AreaID": tool(str(AreaID_batch[0])), "classID": "00" + str(classID)}
        result.append(dict_result)
    with open(result_txt_file, "w") as f:
        for item in result:
            f.write(item["AreaID"] + "\t" + item["classID"] + "\n")






