from data import DataGenerator
from core import ResNet_final

import tensorflow as tf
import datetime
import numpy as np
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
slim = tf.contrib.slim

LOG_DIR = './log_version/'
CHECKPOINT_DIR = './checkpoint_version/'
NUM_CLASSES = 9
BATCHSIZE = 20
LEARNINT_RATE = 0.0001
EPOCHS = 1000
weight_path = "./resnet_first_wights/"

def get_loss(output_concat, onehot):
    with tf.name_scope("loss"):
        # 由于样本不均衡我们添加上class权重
        class_weight = tf.constant([[1, 1.26, 2.64, 7.0, 2.73, 1.72, 2.70, 3.66, 3.30]], dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(tf.cast(onehot, tf.float32), tf.transpose(class_weight)))  # shape [1, batch_size]
        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=output_concat, labels=onehot))  # shape [1, batch_size]
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_concat, labels=onehot)
        cross_entropy_mean = tf.reduce_mean(xent)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.scalar('train_loss', loss)
    return loss


##################### get the input pipline ############################
DataGenerator = DataGenerator.DataGenerator()
TrainDataset = DataGenerator.get_batch(BATCHSIZE, tag="training")
EvalDataset = DataGenerator.get_batch(BATCHSIZE, tag="evaling")

# get the dataset statistics
trainset_length = 380000
eval_set_length = 20000
print("train_set_length:%d" % trainset_length)
print("eval_set_length:%d" % eval_set_length)


iterator = tf.data.Iterator.from_structure(EvalDataset.output_types, EvalDataset.output_shapes)
next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(TrainDataset)
validation_init_op = iterator.make_initializer(EvalDataset)
##################### get the input pipline ############################


##################### setup the network ################################
x = tf.placeholder(tf.float32, shape=(None, 88, 88, 3))
y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
visit_array = tf.placeholder(tf.float32, shape=(None, 174, 24, 2))

is_training = tf.placeholder('bool', [])
keep_prob = tf.placeholder(tf.float32)


depth = 50    # 可以是50、101、152
ResNetModel = ResNet_final.ResNetModel(is_training, depth, NUM_CLASSES)
with tf.name_scope("ResNet"):
    fc_image = ResNetModel.inference(x)
with tf.name_scope("visit"):
    fc_visit = ResNet_final.visit_network(visit_array, is_training)
net_output = ResNet_final.get_net_output(fc_image=fc_image, fc_visit=fc_visit, classNum=NUM_CLASSES, KEEP_PROB=keep_prob)
prediction = tf.argmax(net_output, 1)
groundtruth = tf.argmax(y, 1)

# 训练操作
with tf.name_scope("train"):
    loss = get_loss(net_output, y)
    train_layers = ["visit_scale1", "visit_scale2", "visit_scale3", "scale1", "scale2", "scale3", "scale4", "scale5", "fc"]
    train_op = ResNetModel.optimize(loss=loss, learning_rate=LEARNINT_RATE, train_layers=train_layers)
# 评价操作
with tf.name_scope("eval"):
    correct_pred = tf.equal(tf.argmax(net_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('train_accuracy', accuracy)
summary_op = tf.summary.merge_all()
# 混淆矩阵
confus_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(net_output, 1), num_classes=NUM_CLASSES, name="con_matrix")
##################### setup the network ################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config=config
with tf.Session(config=config) as sess:
    # initial variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 获取预训练的权重
    #ResNetModel.load_original_weights(weight_path=weight_path, session=sess)
    # 判断有没有checkpoint
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored .....")
    sess.graph.finalize()
    # summary
    train_writer = tf.summary.FileWriter(LOG_DIR + "/train", sess.graph)
    eval_writer = tf.summary.FileWriter(LOG_DIR + "/eval", sess.graph)

    # 训练过程
    print("training start")
    train_batches_of_epoch = int(math.ceil(trainset_length/BATCHSIZE))
    for epoch in range(EPOCHS):
        sess.run(training_init_op)
        print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
        step = 1
        while step <= train_batches_of_epoch:
            img_batch, visit_batch, label_batch = sess.run(next_batch)
            #visit1_batch = np.tile(visit1_batch, (1, 1, 8))
            #visit1_batch = np.reshape(visit1_batch, (-1, 182, 192, 1))
            #visit1_batch = np.tile(visit1_batch, (1, 1, 1, 3))
            pre, true, _, loss_value, merge, accu = sess.run([prediction, groundtruth, train_op, loss, summary_op, accuracy], feed_dict={x: img_batch, y: label_batch, is_training: True, visit_array: visit_batch, keep_prob: 0.5})
            #print("{} {} loss = {:.4f}".format(datetime.datetime.now(), step, loss_value))
            #print("accuracy{}".format(accu))
            #print(pre)
            #print(true)
            if step % 20 == 0:
                print("{} {} loss = {:.4f}".format(datetime.datetime.now(), step, loss_value))
                print("accuracy{}".format(accu))
                #saver.save(sess, CHECKPOINT_DIR + "model.ckpt", step)
                #train_writer.add_summary(merge, epoch * train_batches_of_epoch + step)
                #print("checkpoint saved")
            step = step + 1
        saver.save(sess, CHECKPOINT_DIR + "model.ckpt", epoch + 1)
        print("checkpoint saved")
        train_writer.add_summary(merge, epoch + 1)

        # 验证过程
        print("{} Start validation".format(datetime.datetime.now()))
        test_acc = 0.0
        test_count = 0
        eval_batches_of_epoch = int(math.ceil(eval_set_length/BATCHSIZE))

        sess.run(validation_init_op)
        con_mat = np.ones((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
        for tag in range(eval_batches_of_epoch):
            img_batch, visit_batch, label_batch = sess.run(next_batch)
            #visit1_batch = np.tile(visit1_batch, (1, 1, 8))
            #visit1_batch = np.reshape(visit1_batch, (-1, 182, 192, 1))
            #visit1_batch = np.tile(visit1_batch, (1, 1, 1, 3))
            pre, true, acc, con_matrix = sess.run([prediction, groundtruth, accuracy,  confus_matrix], feed_dict={x: img_batch, y: label_batch, is_training: False, visit_array: visit_batch, keep_prob: 1.0})
            con_mat = con_mat + con_matrix
            test_acc += acc
            test_count += 1
            #print("the {} time Validation Accuracy = {:.4f}".format(tag, acc))
            #print(pre)
            #print(true)
            #print(con_mat)
        print(con_mat)
        test_acc /= test_count
        s = tf.Summary(value=[
            tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
        ])
        eval_writer.add_summary(s, epoch + 1)
        print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
