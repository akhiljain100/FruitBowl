#!/usr/bin/env python
import matplotlib.pyplot as plt
import cv2
from threading import Thread, Lock
import numpy as np
from generate_feature import fruit_feature
from song_player import Song

class WebcamVideoStream :
    def __init__(self, src = 1, width = 640, height = 480) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        #self.stdscr = curses.initscr()

    def start(self) :
        if self.started :
            print "already started!!"
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()

            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


class detect_fruit :
    def __init__(self,model_path,read_image,features_path) :
        # Initializing the model
        self.model_path = model_path
        self.model = fruit_feature(self.model_path)
        self.read_image = read_image
        # Objects subject to selection
        self.song = Song(features_path)
        self.songs_feature = self.song.get_song_feature()
        self.start_detecting = False
        self.read_lock = Lock()
        self.detected_feature = None
        self.last_frame = None
    def start(self) :
        if self.start_detecting :
            print "already started!!"
            return None
        self.start_detecting = True
        self.thread = Thread(target=self.detect, args=())
        self.thread.start()
        return self

    def detect(self) :
        while self.start_detecting :

            frame = self.read_image.read()
            if(self.last_frame is not None):
                diff = cv2.absdiff(frame, self.last_frame)
                mean_diff = float(np.mean(diff))


                if mean_diff < 120:
                    self.read_lock.acquire()

                    self.detected_feature = self.model.detect_image(frame)
                    self.last_frame = frame
                    self.read_lock.release()
                else:
                    print('Environment changed, discarding the image')
            else:
                self.last_frame = self.read_image.read()

    def read_feature(self):
        #self.read_lock.acquire()
        if self.detected_feature is None:
            return None
        else:
            detected_feature = self.detected_feature.copy()
            return detected_feature
        #self.read_lock.release()

    def stop(self) :
        self.start_detecting = False
        self.thread.join()

