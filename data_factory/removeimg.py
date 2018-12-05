#coding:utf-8
import os

'''
该程序为当标注结束后,删除未标注的图片
'''
def lll(dirl):
    zhongjian = []
    labelfilelist = os.listdir(dirl)
    for labelfile in labelfilelist:
        labelfile = labelfile.split('.')[0]
        # print(labelfile)
        labelfilelist = zhongjian.append(labelfile)
        # print(zhongjian)
    return zhongjian

def delete(labelfilelist, delPngFloder):
    # labelfilelist = lll(label_path)
    # delPngFloder = del_img_path #删除该文件夹内的未标注图片
    imgfilelist = os.listdir(delPngFloder)
    for imgfile in imgfilelist:
        imgfile = imgfile.split('.')[0]
        if imgfile in labelfilelist:
            pass
        else:
            #print(imgfile)
            delpngPath = delPngFloder + '/' + str(imgfile) + '.jpg'#删除未标注png/jpg
            #delpngPath = delPngFloder + '/' + str(imgfile) + '.txt'#删除多余的txt
            try:
                os.remove(delpngPath)
            except:
                print ('error')
            print (delpngPath)



# label_path   = './label-rec-2596'          
# #待删除png或label的路径    
# del_img_path = './jpg-2596'

# 保留文件的列表
persist_file      = './images-cut/obstacle_train.txt'
#待删除png或label的路径    
del_img_path      = './images-cut/obstacle_train'

idx = []
with open(persist_file) as f:
  for line in f:
    idx.append(line.strip())
f.close()

delete(idx, del_img_path)
print ('delete ok...')
