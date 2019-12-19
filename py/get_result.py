# _*_ coding : UTF-8 _*_
# author : cfl
# time   : 2019/12/18 下午7:23
import face_recognition
import cv2
import numpy as np
from time import sleep


def get_model(path):
    """
    获取存储在本地的模型
    @param path: 项目根目录
    @return: face_encodings_known_ndarray, face_names_known_ndarray
    """
    # 模型路径
    path_model = path + 'model/'

    # 获取面部特征编码列表
    face_encodings_known_ndarray = np.load(path_model + 'model_en.npy')

    # 获取人物姓名列表
    face_names_known_ndarray = np.load(path_model + 'model_na.npy')

    return face_encodings_known_ndarray, face_names_known_ndarray


def get_result(face_encodings_known_ndarray, face_names_known_ndarray, picture=None):
    """
    通过人脸识别算法，获取识别到的人脸，返回人名
    @param picture: 被识别图片路径
    @param face_encodings_known_ndarray: 人脸编码库矩阵
    @param face_names_known_ndarray: 人脸名库矩阵
    @return: name_list
    """
    # 识别到的名字列表
    name_list = []

    # 识别到的人脸
    face_names = []

    # 人脸位置
    face_locations = []

    # 该视频帧状态
    process_this_frame = True

    # 调用摄像头来识别人脸
    video_capture = cv2.VideoCapture(0)

    # 获取目标图片
    # picture = cv2.imread(picture)

    while True:
        # 抓取一帧视频
        ret, frame = video_capture.read()

        # 将图片设置为一帧
        # frame = picture

        # 将视频帧的大小调整为1/4以加快人脸识别处理
        # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
        rgb_small_frame = frame[:, :, ::-1]

        # 仅每隔一帧处理一次视频以节省时间
        # 查找当前视频帧中的所有人脸位置和人脸编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for k in face_encodings:
            # 设置默认名
            name = "Unknown"
            # 查看该人脸是否与已知人脸匹配
            matches = face_recognition.compare_faces(face_encodings_known_ndarray, k)

            # 如果在已知的面编码中找到匹配项，请使用第一个
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # 或者，使用与新人脸的距离最小的已知人脸
            face_distances = face_recognition.face_distance(face_encodings_known_ndarray, k)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = face_names_known_ndarray[best_match_index]

            face_names.append(name)

        # 展示结果
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 由于我们在检测过程中帧被缩放到1/4大小，因此放大备份面位置
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # 在脸上画一个方框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 在人脸下画一个有名字的标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # 显示结果图像
        # cv2.namedWindow('Picture', 0)
        cv2.imshow('Picture', frame)

        # 按键盘上的“q”键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for k in face_names:
        if k not in name_list:
            if k is not "Unknown":
                name_list.append(k)

    # 释放摄像头
    video_capture.release()
    cv2.destroyAllWindows()

    return name_list


if __name__ == '__main__':
    # 项目根路径
    my_path = '../'

    # 获取sample
    my_known_encodings, my_known_names = get_model(my_path)

    # # 全部人脸
    # my_names_list_i = []
    #
    # # 识别文件夹中所有图片
    # dir_list = os.listdir(my_path + "test")
    # for i in dir_list:
    #     my_name = re.findall(r'(\w+)\.', i)[0]
    #     my_picture = my_path + "test/" + i
    #     my_name_list = get_result(my_known_encodings, my_known_names, my_picture)
    #
    #     my_names_list_j = []
    #     for j in my_name_list:
    #         my_names_list_j.append(j)
    #     my_names_list_i.append(my_names_list_j)
    # print(my_names_list_i)

    my_name_list = get_result(my_known_encodings, my_known_names)
    print(my_name_list)
