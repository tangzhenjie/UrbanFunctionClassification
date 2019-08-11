# 这一节我们使用tfrecord来实现读取数据
import tensorflow as tf

tfrecord_dir = "./Dataset/tfrecord/"
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
NUM_CLASSES = 9             # 数据集类别数用于生成one_hot数据
#########生成feature方法##########
def _tf_record_parser(record):
    keys_to_features = {
        'data': tf.FixedLenFeature([], tf.string),
        'visit': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(features['data'], tf.uint8)    # 保存时是uint8所以这必须是uint8(位数必须一样否则报错)
    image = tf.reshape(image, [100, 100, 3])
    image = tf.random_crop(image, [88, 88, 3])  # 随机裁剪
    image = tf.image.random_flip_left_right(image)  # 随机左右翻转
    image = tf.image.random_flip_up_down(image)  # 随机上下翻转
    img_centered = tf.cast(image, tf.float32)
    #img_centered = tf.subtract(image, IMAGENET_MEAN)
    # 在这里可以对图像进行处理（现在我们暂且不处理）
    visit = tf.decode_raw(features['visit'], tf.int32)   # 保存时是int64所以这必须是64位
    visit = tf.reshape(visit, [174, 24, 2])
    visit = tf.cast(visit, tf.float32)

    # 可以在这里改变visit特征的形状使用tf.reshape()
    label = tf.cast(features['label'], tf.int64)
    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return img_centered, visit, label_onehot

def _tf_record_parser_test(record):
    keys_to_features = {
        'data': tf.FixedLenFeature([], tf.string),
        'visit': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(features['data'], tf.uint8)  # 保存时是uint8所以这必须是uint8(位数必须一样否则报错)
    image = tf.reshape(image, [100, 100, 3])
    image = tf.random_crop(image, [88, 88, 3])  # 随机裁剪
    image = tf.image.random_flip_left_right(image)  # 随机左右翻转
    image = tf.image.random_flip_up_down(image)  # 随机上下翻转
    img_centered = tf.cast(image, tf.float32)
    # img_centered = tf.subtract(image, IMAGENET_MEAN)
    # 在这里可以对图像进行处理（现在我们暂且不处理）
    visit = tf.decode_raw(features['visit'], tf.int32)  # 保存时是int64所以这必须是64位
    visit = tf.reshape(visit, [174, 24, 2])
    visit = tf.cast(visit, tf.float32)

    # 可以在这里改变visit特征的形状使用tf.reshape()
    label = tf.cast(features['label'], tf.int64)
    return img_centered, visit, label

#########生成feature方法##########
class DataGenerator(object):
    def __init__(self):
        train_tfrecord_filename = "train.tfrecord"
        eval_tfrecord_filename = "eval.tfrecord"
        test_tfrecord_filename = "test.tfrecord"
        self.train_tfrecord_path = tfrecord_dir + train_tfrecord_filename
        self.eval_tfrecord_path = tfrecord_dir + eval_tfrecord_filename
        self.test_tfrecord_path = tfrecord_dir + test_tfrecord_filename

    def get_batch(self, batch_size, tag="training"):
        if tag == "training":
            training_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            training_dataset = training_dataset.map(_tf_record_parser)
            training_dataset = training_dataset.repeat(None)
            training_dataset = training_dataset.shuffle(buffer_size=5000)
            training_dataset = training_dataset.batch(batch_size)

            return training_dataset
        if tag == "evaling":
            evaling_dataset = tf.data.TFRecordDataset(self.eval_tfrecord_path)
            evaling_dataset = evaling_dataset.map(_tf_record_parser)
            evaling_dataset = evaling_dataset.repeat(1)
            #evaling_dataset = evaling_dataset.shuffle(buffer_size=500)
            evaling_dataset = evaling_dataset.batch(batch_size)
            return evaling_dataset
        if tag == "testing":
            testing_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
            testing_dataset = testing_dataset.map(_tf_record_parser_test)
            testing_dataset = testing_dataset.repeat(1)
            testing_dataset = testing_dataset.batch(batch_size)
            return testing_dataset
