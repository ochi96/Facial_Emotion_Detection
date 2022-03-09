import os
import sys
import cv2
import numpy as np
import time
import tensorflow as tf
import logging
from typing import Sequence, Tuple, Union

import math
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

NumpyRects = Union[np.ndarray, Sequence[Tuple[int, int, int, int]]]
PADDING = 5
logging.basicConfig(level=logging.DEBUG,filename="info.log", filemode='w')

class FER():

    """
    Allows performing Facial Expression Recognition ->
        a) Detection of faces
        b) Detection of emotions
    """

    def __init__(self, emotion_model: str = None, scale_factor: float = 1.1,
                        min_face_size: int = 50, offsets: tuple = (10, 10), compile: bool = False) -> None:

        self.__offsets = offsets
        # Local Keras model
        self.__emotion_classifier = tf.lite.Interpreter(base_dir + 'Models/production_fer_model.tflite')
        self.__emotion_classifier.allocate_tensors()
        self.__emotion_target_size = (64,64)

    @staticmethod
    def pad(image):
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(
            image,
            top = PADDING,
            bottom = PADDING,
            left = PADDING,
            right= PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image

    @staticmethod
    def depad(image):
        row, col = image.shape[:2]
        return image[PADDING : row - PADDING, PADDING : col - PADDING]

    @staticmethod
    def tosquare(bbox):
        """Convert bounding box to square by elongating shorter side."""
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        # if w != h:
        # log.debug(f"{w} is not {h}")
        return (x, y, w, h)

    @staticmethod
    def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def __apply_offsets(self, face_coordinates):
        x, y, width, height = face_coordinates
        x_off, y_off = self.__offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    @staticmethod
    def _get_labels():
        return { 0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

    def detect_emotions(self, img: np.ndarray, face_rectangles: NumpyRects = None) -> list:
        """
        Detects bounding boxes from the specified image with ranking of emotions.
        :param img: image to process (BGR or gray)
        :return: list containing all the bounding boxes detected with their emotions.
        """
        # if img is None or not hasattr(img, "shape"):
        #     raise InvalidImage("Image not valid.")

        emotion_labels = self._get_labels()
        img_copy = img.copy()

        if (face_rectangles is None):
            start = time.time()
            with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5, model_selection=1) as face_detection:
                results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
            end = time.time()
            print('Face detection Time: ', end-start)

            h, w = img.shape[0:2]
            # print(h, w)
            face_rectangles = []
            if results.detections is None:
                return None

            for detection in results.detections:
                print('Face detection confidence: ', detection.score[0])
                b_box = detection.location_data.relative_bounding_box
                rel_x, rel_y, rel_h, rel_w = b_box.xmin, b_box.ymin,  b_box.height, b_box.width

                x =  rel_x *w
                y =  rel_y *h
                w_ =  rel_w *w
                h_ =  rel_h *h
                
                face_rectangles.append([int(item) for item in [x, y, h_, w_]])
        
        print(face_rectangles)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_TOZERO)
        gray_img = self.pad(gray_img)

        emotions = []
        for face_coordinates in face_rectangles:
            start = time.time()
            face_coordinates = self.tosquare(face_coordinates)
            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)
            
            # adjust for padding
            x1 += PADDING
            x2 += PADDING
            y1 += PADDING
            y2 += PADDING            
            x1 = np.clip(x1, a_min=0, a_max=None)
            y1 = np.clip(y1, a_min=0, a_max=None)

            gray_face = gray_img[max(0, y1 - PADDING):y2 + PADDING,
                                    max(0, x1 - PADDING):x2 + PADDING]

            gray_face = gray_img[y1:y2, x1:x2]
            # cv2_imshow(gray_face)

            try:
                gray_face = cv2.resize(gray_face, self.__emotion_target_size)
            except Exception as e:
                # log.info("{} resize failed: {}".format(gray_face.shape, e))
                continue
      
            gray_face = self.__preprocess_input(gray_face, True)
            gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)
            
            end = time.time()
            print('Preprocess Time: ', end-start)
            start = time.time()

            self.__emotion_classifier.set_tensor(self.__emotion_classifier.get_input_details()[0]['index'], gray_face)
            self.__emotion_classifier.invoke()
            
            emotion_prediction = self.__emotion_classifier.get_tensor(self.__emotion_classifier.get_output_details()[0]['index'])[0]
            labelled_emotions = {emotion_labels[idx]: round(float(score), 2) for idx, score in enumerate(emotion_prediction)}
            emotions.append(dict(box=face_coordinates, emotions=labelled_emotions))
            
            end = time.time()
            print('Inference Time: ', end-start)

        self.emotions = emotions
        return emotions

    def top_emotion(self, img: np.ndarray, image_path: str)-> Tuple[Union[str,None], Union[float,None]]:
        """Convenience wrapper for `detect_emotions` returning only top emotion for first face in frame.
        :param img: image to process
        :return: top emotion and score (for first face in frame) or (None, None)
        """
        box_emotions = self.detect_emotions(img=img)

        # saving results to .txt file
        file_name = image_path.rsplit('/')[-1]
        print(file_name)
        
        if not box_emotions:
            print('No face detected')
            # with open(base_dir + 'Data/MediapipeGrayWithContrast_part_2_4.3.txt', 'a') as f:
            #     f.write('{0}'.format(file_name))
            #     f.write('No face Detected \n')
            #     f.write('\n')
            return None
        else:
            top_emotions = [max(e["emotions"], key=lambda key: e["emotions"][key]) for e in box_emotions]
            scores = [box_emotions[i]['emotions'][top_emotions[i]] for i in range(len(top_emotions))]

            face_box = [e['box'] for e in box_emotions]
            face_details = list(zip(face_box, top_emotions, scores))
            print(face_details)

            # with open(base_dir + 'Data/MediapipeGrayWithContrast_part_2_4.3.txt', 'a') as f:
            #     f.write('{0} \n'.format(file_name))
            #     for i, face in enumerate(box_emotions):
            #         f.write('Emotions: {} \n'.format(face['emotions']))
            #         f.write('{}: {}'.format(top_emotions, scores))
            #         f.write('\n\n')

        return face_details

def display_results(face_details, image, image_path, font_scale = 0.5, font = cv2.FONT_HERSHEY_SIMPLEX, FONT_COLOR = (0, 0, 0),
                        FONT_THICKNESS = 2, rectangle_pos = (255, 0, 0), rectangle_neg = (0, 0, 255)):
    
    
    positive_emotions = ['happy','surprise', 'neutral']
    output_file_path = base_dir + "Data/Sandbox/{}".format(image_path.rsplit('/')[-1])

    if face_details:
        for face_detail in face_details:
            (x, y, h, w), emotion, confidence = face_detail
            if emotion in positive_emotions:
                text = '(pos){0}: {1}'.format(emotion, confidence)
                cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_pos, 2)
            else:
                text = '(neg){0}: {1}'.format(emotion, confidence)
                cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_neg, 2)

            text_x, text_y = (x, y-10)
            # get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
            # Set the Coordinates of the boxes
            box_coords = ((text_x - 10, text_y + 4), (text_x + text_width + 10, text_y - text_height - 5))
            # Draw the detected boxes and labels
            cv2.rectangle(image, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
            cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)
        cv2.imwrite(output_file_path, image)
    pass



logging.debug('lol lol lol')
logging.info('application started')
logging.info('\n')
logging.info('application started')
start = time.time()

base_dir = './'
image_path = base_dir + "Data/Near/54.png"

print(image_path)
image = cv2.imread(image_path)
# image = cv2.imread(base_dir + "Data/Test/justin.jpg" )
end = time.time()
print('File reading time: ', end-start )
fer = FER()
face_details = fer.top_emotion(image, image_path)
display_results(face_details, image, image_path)



