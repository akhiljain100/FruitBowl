# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np




video_capture = cv2.VideoCapture(-1)
count  = 0
while(True):

    rubbish, original_frame = video_capture.read()
    #original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    cv2.waitKey(3000)

    cv2.imwrite("frame%d.jpg"% count, original_frame)
    count += 1
    if (count == 10):
        break
    print(count)


rubbish, original_frame = video_capture.read()

#plt.imshow(original_frame)
#plt.show()
