import cv2
import sys
import logging as log
import datetime as dt
import time
import os
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
pre_frame = None  # 总是取视频流前一帧做为背景相对下一帧进行比较
fps = 5  # 帧率
count_moving = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    start = time.time()
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
	    break
    end = time.time()

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Bryan\' Web Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Bryan\' Web Video', frame)
	# 运动检测部分
    seconds = end - start
    if seconds < 1.0 / fps:
        time.sleep(1.0 / fps - seconds)
    gray = cv2.resize(gray, (500, 500))
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 

    # 在完成对帧的灰度转换和平滑后，就可计算与背景帧的差异，并得到一个差分图（different map）。还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    if pre_frame is None:
        pre_frame = gray
    else:
        img_delta = cv2.absdiff(pre_frame, gray)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 5000: # 设置敏感度
                continue
            else:
                count_moving  = count_moving  + 1
                print("Moving counts is ",count_moving)
                os.system(r'"C:\Program Files (x86)\Notepad++\notepad++.exe"')
                break
        pre_frame = gray
	
	
	

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
