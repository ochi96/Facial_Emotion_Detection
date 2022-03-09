#from threading import Thread
import cv2, time, queue, threading
import logging

class VideoStreamWidget(object):

    __SCREEN_WIDTH = 320  
    __SCREEN_HEIGHT = 240

    def __init__(self, src=1):
        self.capture = cv2.VideoCapture(src)
        #Start the thread to read frames from the video stream
        self.capture.set(3, self.__SCREEN_WIDTH)
        self.capture.set(4, self.__SCREEN_HEIGHT)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        '''read frames as soon as they are available, keeping only most recent one'''

        while self.capture.isOpened():
            i=0
            while i >= 0:
                (success, self.frame) = self.capture.read()
                # fps = self.capture.get(cv2.CAP_PROP_FPS)
                # print(fps)
                if not success:
                    break
                if not self.q.empty():          #meaning if there is a frame already in the queue
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                i += 1
                self.q.put(self.frame)   #adding frames to the queue..... which is only one image 
    
    def read(self):
        return self.q.get()

 
