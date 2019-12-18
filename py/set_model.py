# _*_ coding : UTF-8 _*_
# author : cfl
# time   : 2019/12/18 下午6:31
import face_recognition
import os
import re
import numpy as np


def set_model(path):
    """
    读取样板地址下的图片文件，从而获取模型并保存
    @param path: 项目根目录
    @return:
    """
    # 样板地址
    path_sample = path + 'sample/'
    # 模型地址
    path_model = path + 'model/'

    # 面部特征编码列表
    face_encodings_known_list = []
    # 人物姓名列表
    face_names_known_list = []

    # 获取所有sample文件
    dir_list = os.listdir(path_sample)
    for j in dir_list:
        name = re.findall(r'(\w+)\.', j)[0]
        file_name = path_sample + j

        image = face_recognition.load_image_file(file_name)
        face_encodings_known_list.append(face_recognition.face_encodings(image)[0])
        face_names_known_list.append(name)

    # 保存编码库
    np.save(path_model + 'model_en.npy', face_encodings_known_list)
    # 保存姓名库
    np.save(path_model + 'model_na.npy', face_names_known_list)

    print("保存sample完成！")


if __name__ == '__main__':
    my_path = '../'

    set_model(my_path)
