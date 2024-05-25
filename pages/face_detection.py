import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
import joblib
import os

st.set_page_config(page_title="Nhận diện khuôn mặt", page_icon="😃", layout="wide")

st.markdown("# Nhận diện khuôn mặt")
FRAME_WINDOW = st.image([])

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

# Lấy đường dẫn tuyệt đối của thư mục chứa file script hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Xây dựng đường dẫn đến thư mục chứa ảnh 'stop.jpg'
image_dir = os.path.join(current_dir, '..', 'module', 'nhan_dien_khuon_mat')

svc = joblib.load('module/nhan_dien_khuon_mat/svc.pkl')
mydict = ['An', 'HocTuan', 'Huy', 'Toan', 'Trung']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        'module/nhan_dien_khuon_mat/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
        'module/nhan_dien_khuon_mat/face_recognition_sface_2021dec.onnx', "")

    tm = cv.TickMeter()

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        video_path = os.path.join(current_dir, 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Open the video file
        cap = cv.VideoCapture(video_path)
    else:
        # Use webcam if no video file is uploaded
        cap = cv.VideoCapture(0)

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            st.error('Không thể nhận hình ảnh!')
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()
        
        if faces[1] is not None:
            for face in faces[1]:
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                coords = face[:-1].astype(np.int32)
                cv.putText(frame, result, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        visualize(frame, faces, tm.getFPS())

        FRAME_WINDOW.image(frame, channels='BGR')

    cap.release()
    cv.destroyWindow()
