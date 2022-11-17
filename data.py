import nibabel
import scipy.io as sio
import scipy.io as scio
from PIL import Image
import numpy as np
import os

from matplotlib import pyplot


def mat_nii():

    data = sio.loadmat('DCE_IMs.mat')  # 加载原始mat
    nibabel.Nifti1Image(data['IMs'], None).to_filename('DCE_IMs.nii.gz')  # 转化成nii写入磁盘

    img = nibabel.load('DCE_IMs.nii.gz')  # 从磁盘读文件
    data = img.get_data()  # [128,128,247]


def mat_png():
    path = './images/'
    data = sio.loadmat('DCE_IMs.mat')
    ims = data['IMs']
    pyplot.imshow(ims[:,:,0])
    pyplot.show()
    for img_id in range(247):
        im = ims[:, :, img_id] /430 * 255
        new_im = Image.fromarray(im.astype(np.uint8))
        new_im.save(path+'ims'+str(img_id)+ '.png')  # 保存图片


def divide_mat():
    data = sio.loadmat('DCE_IMs.mat')
    ims = data['IMs']
    for img_id in range(247):
        name = './mat/ims'+str(img_id)+'.mat'
        scio.savemat(name, {'IMs': ims[:, :, img_id]})


if __name__ == '__main__':
    # mat_nii()
    # mat_png()
    divide_mat()
