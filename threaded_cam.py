import cv2, queue, threading

class VideoStreamWidget(object):

    def __init__(self, src=0, SCREEN_WIDTH=400, SCREEN_HEIGHT=600):

        self.capture = cv2.VideoCapture(src)
        self.__SCREEN_WIDTH = SCREEN_WIDTH
        self.__SCREEN_HEIGHT = SCREEN_HEIGHT
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
            while True:
                (success, self.frame) = self.capture.read()
                # fps = self.capture.get(cv2.CAP_PROP_FPS)
                if not success:
                    break
                if not self.q.empty():          #meaning if there is a frame already in the queue
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(self.frame)   #adding frames to the queue..... which is only one image 

    def read(self):
        return self.q.get()

 
