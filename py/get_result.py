# _*_ coding : UTF-8 _*_
# author : cfl
# time   : 2019/12/18 下午5:04
import face_recognition
import cv2
import numpy as np
import os
import re
import pymysql


def get_model(path):
    """
    获取存储在本地的模型
    @param path:
    @return:
    """

    # 获取面部特征编码列表
    face_encodings_known_ndarray = np.load('../' + 'model_en.npy')

    # 获取人物姓名列表
    face_names_known_ndarray = np.load('../' + 'model_na.npy')

    return face_encodings_known_ndarray, face_names_known_ndarray


def get_result(test_picture, known_face_encodings, known_face_names):

    names = []

    face_locations = []
    face_names = []
    process_this_frame = True

    # 调用摄像头来识别人脸
    # Get a reference to webcam #0 (the default one)
    # video_capture = cv2.VideoCapture(0)

    # 获取目标图片
    picture = cv2.imread(test_picture)

    while True:
        # Grab a single frame of video
        # ret, frame = video_capture.read()
        frame = picture

        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            print("识别成功！")

            process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.namedWindow('Picture', 0)
        cv2.imshow('Picture', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for j in face_names:
        if j not in names:
            if j is not "Unknown":
                names.append(j)

    # Release handle to the webcam
    # picture.release()
    cv2.destroyAllWindows()

    return names


if __name__ == '__main__':
    my_path = '../'

    db = pymysql.connect("127.0.0.1", "root", "cfl656359504", "ims")
    cursor = db.cursor()

    # 获取sample
    known_encodings, known_names = get_model(my_path)

    # 识别文件夹中所有图片
    dir_ = os.listdir("../test/")
    for i in dir_:
        id_ = re.findall(r'(\w+)\.', i)[0]
        test = "../test/" + i
        names = get_result(test, known_encodings, known_names)

        names_str = ""
        for i in names:
            names_str = names_str + " " + i
            print(i)

        cursor.execute("insert into face values ('%s', %d, '%s')" % (
        id_, len(names), names_str))

        db.commit()

    # test = "../test/hezhao.jpg"
    # names = get_result(test, known_encodings, known_names)
    #
    # names_str = ""
    # for i in names:
    #     names_str = names_str + " " + i
    #     print(i)
    #
    # cursor.execute("insert into face values ('%s', %d, '%s')" % (time.strftime("%Y%m%d%H%M%S", time.localtime()), len(names), names_str))
    #
    # db.commit()

    db.close()
