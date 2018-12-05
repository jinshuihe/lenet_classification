# -- coding: utf-8 --
# 制作车位类别训练数据

import textwrap
import glob
import os
import cv2  
import numpy as np 
import tensorflow as tf


def load_kitti_annotation(label_path,image_name):#每一个label的坐标
    box = []
    filename = os.path.join(label_path, image_name+'.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()#读取所有行
    f.close()
    for line in lines:#读取每一行
        obj = line.strip().split(',')
        cls = obj[8]
        x0 = int(obj[0])
        y0 = int(obj[1])
        x1 = int(obj[2])
        y1 = int(obj[3])
        x2 = int(obj[4])
        y2 = int(obj[5])
        x3 = int(obj[6])
        y3 = int(obj[7])

        box.append([x0,y0,x1,y1,x2,y2,x3,y3,cls])
    return box
  

#获取文件名列表
def get_file_name_list(file_dir, imgType):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == imgType:
                file = file[0:-4]  
                L.append(file)  
    return L 


# file_folder       = '/home/lwj/DATA_Factory/jpg-2596/' 
# txt_folder        = '/home/lwj/DATA_Factory/label-rec-2596/' 
imgType          = '.png'
file_folder      = '/home/lwj/code/AdvancedEAST/data/images-val/' 
txt_folder       = '/home/lwj/code/AdvancedEAST/data/labels-val/'
file_cls0_folder = './val-images/0/'  
file_cls1_folder = './val-images/1/'  
file_cls2_folder = './val-images/2/'  

file_name_list = get_file_name_list(file_folder, imgType)

if not tf.gfile.Exists(file_cls0_folder):
    tf.gfile.MakeDirs(file_cls0_folder)
if not tf.gfile.Exists(file_cls1_folder):
    tf.gfile.MakeDirs(file_cls1_folder)
if not tf.gfile.Exists(file_cls2_folder):
    tf.gfile.MakeDirs(file_cls2_folder)

for file_name in file_name_list:
    file_path     = os.path.join(file_folder, file_name + imgType)

    img = cv2.imread(file_path) 
    height, width = img.shape[:2]
    four_points_list = load_kitti_annotation(txt_folder, file_name)

    idx_num_cut = 0
    for pts in four_points_list:      
        pt_x = []
        pt_y = []
        for idx in range(4):
            pt_x.append(pts[idx * 2])
            pt_y.append(pts[idx * 2 + 1])
            
        pt_x.sort()
        pt_y.sort()
        # cv2.rectangle(img, (pt_x[0], pt_y[0]), (pt_x[3], pt_y[3]), 255, 1)
        img_cut = img[pt_y[0]:pt_y[3], pt_x[0]:pt_x[3], :] #[height,width]剪切图像

        if   pts[8] == '0':
            store_img_path = os.path.join(file_cls0_folder, file_name + '_' + str(idx_num_cut) + '-cls0.jpg')
            cv2.imwrite(store_img_path, img_cut)
        elif pts[8] == '1':
            store_img_path = os.path.join(file_cls1_folder, file_name + '_' + str(idx_num_cut) + '-cls1.jpg')
            cv2.imwrite(store_img_path, img_cut)
        elif pts[8] == '2':
            store_img_path = os.path.join(file_cls2_folder, file_name + '_' + str(idx_num_cut) + '-cls2.jpg')
            cv2.imwrite(store_img_path, img_cut)

        print (store_img_path)
        idx_num_cut += 1

print ('OK...')