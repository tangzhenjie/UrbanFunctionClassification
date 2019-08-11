from data import DataGenerator
from core import ResNet_final_new10

import tensorflow as tf
import math
import numpy as np
slim = tf.contrib.slim

result_txt_file = "result_version19_new.txt"
CHECKPOINT_DIR = './checkpoint_version_final/'
NUM_CLASSES = 9
BATCHSIZE = 1
def tool(AreaID):
    last = 6 - len(AreaID)
    for key in range(last):
        AreaID  = "0" + AreaID
    return AreaID

##################### get the input pipline ############################
DataGenerator = DataGenerator.DataGenerator()
TestDataset = DataGenerator.get_batch(BATCHSIZE, tag="testing")
# get the dataset statistics
test_set_length = 100000
print("test_set_length:%d" % test_set_length)

iterator = TestDataset.make_one_shot_iterator()
next_batch = iterator.get_next()
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 88, 88, 3))
visit_array1 = tf.placeholder(tf.float32, shape=(None, 174, 24, 2))
#visit_array2 = tf.placeholder(tf.float32, shape=(None, 500))
is_training = tf.placeholder('bool', [])


depth = 50    # 可以是50、101、152
ResNetModel = ResNet_final_new10.ResNetModel(is_training, depth, NUM_CLASSES)
with tf.name_scope("ResNet"):
    fc_image = ResNetModel.inference(x)
with tf.name_scope("visit"):
    fc_visit = ResNet_final_new10.visit_network(visit_array1, is_training)
net_output = ResNet_final_new10.get_net_output(fc_image=fc_image, fc_visit=fc_visit, classNum=NUM_CLASSES, KEEP_PROB=1.0)



# 评价操作
prediction = tf.argmax(net_output, 1)





##################### setup the network ################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
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
    result = np.zeros(shape=[100000], dtype=np.uint8)
    result = []
    for step in range(test_batches_of_epoch):
        img_batch, visit1_batch, AreaID_batch = sess.run(next_batch)
        classIDs = sess.run(prediction, feed_dict={x: img_batch, visit_array1: visit1_batch, is_training: False})
        classID = (classIDs[0] + 1)
        print("AreaID %d" % AreaID_batch[0])
        print(classID)
        dict_result = {"AreaID": tool(str(AreaID_batch[0])), "classID": "00" + str(classID)}
        result.append(dict_result)
    with open(result_txt_file, "w") as f:
        for item in result:
            f.write(item["AreaID"] + "\t" + item["classID"] + "\n")






