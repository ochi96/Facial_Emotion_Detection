import os
import logging
import cv2
from threaded_cam import VideoStreamWidget


class   Livestream():

    def __init__(self):
        self.livefeed = VideoStreamWidget()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter('lol.avi', self.fourcc, 20.0, (320, 240))

    def view(self):
        while True:
            number, frame = self.livefeed.read()
            self.video.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.video.release()
                cv2.destroyAllWindows()
                break

Livestream().view()

