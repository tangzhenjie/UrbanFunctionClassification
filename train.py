from data.DataGenerator import ImageDataGenerator
from core import model

import tensorflow as tf
import datetime
import numpy as np
import math
slim = tf.contrib.slim

LOG_DIR = 'D:\\pycharm_program\\UrbanFunctionClassification\\log\\'
DATASET_DIR = "D:\\competition\\data\\train\\"
CHECKPOINT_DIR = 'D:\\pycharm_program\\UrbanFunctionClassification\\checkpoint\\'
NUM_CLASSES = 9
BATCHSIZE = 20
LEARNINT_RATE = 0.0001

def get_loss(output_concat, onehot):
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=output_concat, labels=onehot)
        loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', loss)
    return loss

def train(loss_val, var_list, learning_rate):
    gradients = tf.gradients(loss_val, var_list)  # 实现loss对训练变量求导，返回一个梯度列表
    gradients = list(zip(gradients, var_list))  # zip即打包为元组，并将元组转为列表
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    return train_op

##################### get the input pipline ############################
DataGenerator = ImageDataGenerator(DATASET_DIR, mode="training")
# get the dataset statistics
trainset_length = DataGenerator.train_set_length
eval_set_length = DataGenerator.eval_set_length
print("train_set_length:%d" % trainset_length)
print("eval_set_length:%d" % eval_set_length)

TrainDataset = DataGenerator.getBatchData(batch_size=BATCHSIZE, num_classes=NUM_CLASSES)
EvalDataset = DataGenerator.getBatchData(batch_size=BATCHSIZE, num_classes=NUM_CLASSES, shuffle=False)

iterator = tf.data.Iterator.from_structure(TrainDataset.output_types, TrainDataset.output_shapes)
next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(TrainDataset)
validation_init_op = iterator.make_initializer(EvalDataset)
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 100, 100, 3))
y = tf.placeholder(tf.int32, shape=(None, NUM_CLASSES))

with slim.arg_scope(model.vgg_arg_scope()):
    net_output, end_points = model.vgg_16(x, NUM_CLASSES)


# 训练操作
with tf.name_scope("train"):
    loss = get_loss(net_output, y)
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var, LEARNINT_RATE)
# 评价操作
with tf.name_scope("eval"):
    correct_pred = tf.equal(tf.argmax(net_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
summary_op = tf.summary.merge_all()
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

    # summary
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train", sess.graph)

    # 训练过程
    print("training start")
    sess.run(training_init_op)
    train_batches_of_epoch = int(math.ceil(trainset_length/BATCHSIZE))
    for step in range(51):
        img_batch, label_batch = sess.run(next_batch)
        pre, true, _, loss_value, merge, accu = sess.run([tf.argmax(net_output, 1), tf.argmax(y, 1), train_op, loss, summary_op, accuracy], feed_dict={x: img_batch, y: label_batch})
        print("{} {} loss = {:.4f}".format(datetime.datetime.now(),step + 1, loss_value))
        print("accuracy{}".format(accu))
        print(pre)
        print(true)
        if step % 50 == 0:
            saver.save(sess, CHECKPOINT_DIR + "model.ckpt", step)
            train_writer.add_summary(merge, step)
            print("checkpoint saved")


    # 验证过程
    print("eval start")
    test_acc = 0.0
    test_count = 0
    eval_batches_of_epoch = int(math.ceil(eval_set_length/BATCHSIZE))

    sess.run(validation_init_op)
    for tag in range(eval_batches_of_epoch):
        img_batch, label_batch = sess.run(next_batch)
        pre, true, acc = sess.run([tf.argmax(net_output, 1), tf.argmax(y, 1), accuracy], feed_dict={x: img_batch, y: label_batch})
        test_acc += acc
        test_count += 1
        print("the {} time Validation Accuracy = {:.4f}".format(tag, acc))
        print(pre)
        print(true)
    test_acc /= test_count
    print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
