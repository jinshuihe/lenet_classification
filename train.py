
# -- coding: utf-8 --


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import infer
import os
import numpy as np

import cv2
import glob
import random
from tensorflow.python.platform import gfile


##### 1. 定义神经网络相关的参数

VALIDATION_PERCENTAGE  = 10

BATCH_SIZE             = 16
LEARNING_RATE_BASE     = 0.01
LEARNING_RATE_DECAY    = 0.99
REGULARIZATION_RATE    = 0.001
TRAINING_STEPS         = 1000000
MOVING_AVERAGE_DECAY   = 0.99

train_dir              = "./log-1204-1"
CACHE_DIR              = './data_factory/neck'
INPUT_DATA             = './data_factory/train-images'


# 把样本中所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(validation_percentage):

    # 所有图片存在result字典中,key为类别名
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下的所有有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()
        
        # 初始化当前类别的训练\验证集
        training_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据存入结果字典
        result[label_name] = {  'dir': dir_name,
                                'training': training_images,
                                'validation': validation_images}
    return result


# 定义函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 定义函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def read_img_data(sess, image_lists, label_name, index, category):  
    # 获取一张图片对应的特征向量文件的路径  
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)

    img = cv2.imread(image_path)
    img = img.astype(np.float32, copy=False)
    img = cv2.resize(img, (infer.IMAGE_SIZE, infer.IMAGE_SIZE))
    img = img[:, :, 0] #B
    img = img/256 #(0,1) 归一化

    return img

# 随机获取一个batch的图片作为训练数据。
def get_random_cached_image_jpgs(sess, n_classes, image_lists, how_many, category):

    cls_train_num = []
    for index in range(n_classes):
        cls_train_num.append(len(image_lists[str(index)]['training']))

    image_jpgs = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes) #随机取编号
        image_index = random.randrange(cls_train_num[label_index])
        # bottleneck 
        image_jpg = read_img_data(sess, image_lists, str(label_index), image_index, category)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        image_jpgs.append(image_jpg)
        ground_truths.append(ground_truth)

    return image_jpgs, ground_truths


def train():

    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir

    image_lists = create_image_lists(VALIDATION_PERCENTAGE)

    n_classes = infer.NUM_LABELS

    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    infer.IMAGE_SIZE,
                                    infer.IMAGE_SIZE,
                                    infer.NUM_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, infer.NUM_LABELS], name='y-input')

    # # 显示图片信息
    with tf.name_scope('input_reshape'):###
        image_shaped_input = tf.reshape(x, [-1, infer.IMAGE_SIZE, infer.IMAGE_SIZE, infer.NUM_CHANNELS])
        # tf.summary.image('input', image_shaped_input, 3)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = infer.inference(x, False, regularizer)#推理输出
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # 交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = LEARNING_RATE_BASE

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    tf.summary.image('input', image_shaped_input, 3)     
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    # 将所有日志 tf.summary. 写入文件
    merged = tf.summary.merge_all()
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):

            train_image_jpgs, train_ground_truth = get_random_cached_image_jpgs(
                sess, n_classes, image_lists, BATCH_SIZE, 'training')

            reshaped_xs = np.reshape(train_image_jpgs, 
                                      ( BATCH_SIZE,
                                        infer.IMAGE_SIZE,
                                        infer.IMAGE_SIZE,
                                        infer.NUM_CHANNELS))

            summary, _, loss_value, step = sess.run([merged, 
                                                     train_op, 
                                                     loss, 
                                                     global_step], 
                                                    feed_dict={x: reshaped_xs, 
                                                               y_: train_ground_truth})   

            if i % 200 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
                if i % 4000 == 0:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt'))                
                    saver.save(sess, checkpoint_path, global_step=i)

                tf.summary.scalar('loss_value', loss_value)
                summary_writer.add_summary(summary, i)

        summary_writer.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    main()

