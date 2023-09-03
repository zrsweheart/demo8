import os
import shutil

import SimpleITK as sitk

import numpy as np


def nii2dcm_single(nii_path, IsData = True):
    save_folder = './dcm/dcm_single'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
    if IsData:  # 过滤掉其他无关的组织，标签不需要这步骤
        data1[data1 > 250] = 250
        data1[data1 < -250] = -250
    img_name = os.path.split(nii_path)  #分离文件名
    img_name = img_name[-1]
    img_name = img_name.split('.')
    img_name = img_name[0]
    i = data1.shape[0]
    # 关键部分
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)
    for j in range(0, i):   #将每一张切片都转为png
        if IsData:  # 数据
            slice_i = data1[j, :, :]
            data_img = sitk.GetImageFromArray(slice_i)
            # Convert floating type image (imgSmooth) to int type (imgFiltered)
            data_img = castFilter.Execute(data_img)
            sitk.WriteImage(data_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))
        else:   # 标签
            slice_i = data1[j, :, :] * 122
            label_img = sitk.GetImageFromArray(slice_i)
            # Convert floating type image (imgSmooth) to int type (imgFiltered)
            label_img = castFilter.Execute(label_img)
            sitk.WriteImage(label_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))


def nii2dcm(data_path, lable_path, IsTrain = False):
    save_folder = './dcms'
    nii_list = os.listdir(lable_path)
    for nii_label in nii_list:
        print("process:", nii_label)

        data1 = sitk.ReadImage(os.path.join(lable_path, nii_label)) # 读取一个数据
        liver_tumor_label = sitk.GetArrayFromImage(data1)  # 获取数据的array
        print(np.unique(liver_tumor_label))
        data2 = sitk.ReadImage(os.path.join(data_path, nii_label.replace('segmentation', 'volume')))  # 读取一个数据
        liver_tumor_data = sitk.GetArrayFromImage(data2)  # 获取数据的array
        slice_num = liver_tumor_label.shape[0]     # 获取切片数量
        if IsTrain:
            save_img = save_folder+'/train'
        else:
            save_img = save_folder + '/val'
        label_path = save_folder+'/label'     # 设置保存路径
        if not os.path.exists(save_img):      # 判断文件目录是否存在，不存在则创建
            os.makedirs(save_img)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        img_name = os.path.split(nii_label.replace('segmentation', 'volume'))     # 分离文件名，用以保存时的文件名前缀
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0]

        # 前者为路径，后者为待创建的文件夹名称
        dir = os.path.join(r'dcm', img_name)

        # 判断上述路径中是否已经存在目标文件夹
        isExists = os.path.exists(dir)

        # 如果不存在则创建，否则不做任何操作
        if not isExists:
            os.mkdir(dir)

        for i in range(0, slice_num):
            # 关键部分
            if liver_tumor_label[i, :, :].max() > 0:    # 肝脏标签图片
                label_img = sitk.GetImageFromArray(liver_tumor_label[i, :, :] * 122)
                data_img = sitk.GetImageFromArray(liver_tumor_data[i, :, :])
                castFilter = sitk.CastImageFilter()
                castFilter.SetOutputPixelType(sitk.sitkInt16)
                # Convert floating type image (imgSmooth) to int type (imgFiltered)
                label_img = castFilter.Execute(label_img)
                data_img = castFilter.Execute(data_img)
                sitk.WriteImage(label_img, "%s/%s-%d.dcm" % (label_path, img_name, i))
                shutil.move( "%s/%s-%d.dcm" % (label_path, img_name, i), "%s/%s" % (r'dcm', img_name))
                sitk.WriteImage(data_img, "%s/%s-%d.dcm" % (save_img, img_name, i))








if __name__ == "__main__":
    # 单个nii文件转png
    #nii_single = r'G:\Python\liversegment\ImageResource\ImageTraning\data\volume-0.nii'

    niipath = r'data'
    labpath = r'label'


    # 单个nii转dcm
   # nii2dcm_single(nii_single, True)
    # 批量nii转dcm
    nii2dcm(niipath, labpath, IsTrain=True)


