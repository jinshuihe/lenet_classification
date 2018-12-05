
# -- coding: utf-8 --

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import infer
import os
import numpy as np
import cv2


BATCH_SIZE = 1
REGULARIZATION_RATE = 0.0001
model_path          = './log-1204-1/model.ckpt-988000'

test_data_folder  = ['./data_factory/val-images/0',
                     './data_factory/val-images/1',
                     './data_factory/val-images/2']


def get_path_list(file_dir):   
    L=[]   
    for folder in file_dir:
        for root, dirs, files in os.walk(folder):  
            for file in files:  
                if os.path.splitext(file)[1] == '.jpg':  
                    L.append(os.path.join(root, file))   # path and label
    return L  

def test():
    with tf.get_default_graph().as_default():

        x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                        infer.IMAGE_SIZE,
                                        infer.IMAGE_SIZE,
                                        infer.NUM_CHANNELS],
                           name='x-input')

        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        y = infer.inference(x, False, regularizer)
        y_cls = tf.argmax(tf.nn.softmax(y), 1)
    
        saver = tf.train.Saver()

        with tf.Session() as sess:
            
            saver.restore(sess, model_path)

            numOK = 0
            numNG = 0

            data_list = get_path_list(test_data_folder)

            for img_path in data_list:

                imgOri = cv2.imread(img_path)  
                img    = cv2.resize(imgOri, (infer.IMAGE_SIZE, infer.IMAGE_SIZE))
                img    = img[:, :, 0] #B
                img    = img/256 

                reshaped_xs = np.reshape(img, ( BATCH_SIZE,
                                                infer.IMAGE_SIZE,
                                                infer.IMAGE_SIZE,
                                                infer.NUM_CHANNELS))
                                            
                out_y_cls = sess.run([y_cls], feed_dict={x: reshaped_xs})

                cls_out = out_y_cls[0][0]                           
                
                cls_gt = int(img_path[-5:-4]) #根据文件名获取类别标签

                if cls_gt == cls_out:
                    numOK += 1
                elif cls_gt == 0 and cls_out == 2:
                    numOK += 1
                elif cls_gt == 2 and cls_out == 0:
                    numOK += 1
                else:
                    numNG += 1
                    print('{}     {}, {}'.format(img_path, cls_gt, cls_out))

                    folder, imgName = os.path.split(img_path)
                    outFolder = os.path.join('./out/' + str(cls_gt) + '/' + str(cls_out) + '/')
                    if not os.path.exists(outFolder):
                        os.makedirs(outFolder)
                    cv2.imwrite(os.path.join(outFolder + imgName + '.jpg'), imgOri)


            print('precision:', float(numOK)/float(numOK + numNG))

test()


