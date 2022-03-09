import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG,filename="info.log", filemode='w')

class FER():

    """
    Allows performing Facial Expression Recognition ->
        a) Detection of faces
        b) Detection of emotions
    """

    def __init__(self) -> None:
        self.__offsets = (10, 10)
        self.__emotion_classifier = tf.lite.Interpreter('./Models/production_fer_model.tflite')
        self.__emotion_classifier.allocate_tensors()
        self.__emotion_target_size = (64,64)
        self.__padding = 5
        self.__mp_face_detection = mp.solutions.face_detection

    def __pad(self, image):
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(image,
            top = self.__padding,
            bottom = self.__padding,
            left = self.__padding,
            right= self.__padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image
    
    @staticmethod
    def _get_labels():
        return { 0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

    @staticmethod
    def __tosquare(bbox):
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
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

    def __facepad(self, image, detection):

        h, w = image.shape[0:2]

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, gray_img = cv2.threshold(gray_img, 30, 255, cv2.THRESH_TOZERO)
        gray_img = self.__pad(gray_img)

        b_box = detection.location_data.relative_bounding_box
        rel_x, rel_y, rel_h, rel_w = b_box.xmin, b_box.ymin,  b_box.height, b_box.width
        x =  rel_x *w
        y =  rel_y *h
        w_ =  rel_w *w
        h_ =  rel_h *h

        face_coordinates = [int(item) for item in [x, y, h_, w_]]
        face_coordinates = self.__tosquare(face_coordinates)
        x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)
        
        # adjust for padding
        x1 += self.__padding
        x2 += self.__padding
        y1 += self.__padding
        y2 += self.__padding
        x1 = np.clip(x1, a_min=0, a_max=None)
        y1 = np.clip(y1, a_min=0, a_max=None)

        gray_face = gray_img[max(0, y1 - self.__padding):y2 + self.__padding, max(0, x1 - self.__padding):x2 + self.__padding]
        gray_face = gray_img[y1:y2, x1:x2]
        
        gray_face = cv2.resize(gray_face, self.__emotion_target_size)
        gray_face = self.__preprocess_input(gray_face, True)
        gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)

        return face_coordinates, gray_face
    
    def predict_emotion(self, gray_face):

        emotion_labels = self._get_labels()
        self.__emotion_classifier.set_tensor(self.__emotion_classifier.get_input_details()[0]['index'], gray_face)
        self.__emotion_classifier.invoke()
        
        emotion_prediction = self.__emotion_classifier.get_tensor(self.__emotion_classifier.get_output_details()[0]['index'])[0]
        labelled_emotions = {emotion_labels[idx]: round(float(score), 2) for idx, score in enumerate(emotion_prediction)}

        emotion = max(labelled_emotions, key=labelled_emotions.get)
        score = max(labelled_emotions.values())

        return emotion, score
    
    @staticmethod
    def display_results(image, face_coordinates, emotion, score, font_scale = 0.5, font = cv2.FONT_HERSHEY_SIMPLEX, FONT_COLOR = (0, 0, 0),
                        FONT_THICKNESS = 2, rectangle_pos = (255, 0, 0), rectangle_neg = (0, 0, 255) ):
        text = '{0}: {1}'.format(emotion, score)
        x, y, w, h = face_coordinates
        positive_emotions = ['happy','surprise', 'neutral']
        if emotion in positive_emotions:
            cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_pos, 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_neg, 2)
        
        text_x, text_y = (x, y-10)
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
        box_coords = ((text_x - 10, text_y + 4), (text_x + text_width + 10, text_y - text_height - 5))
        cv2.rectangle(image, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
        cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)

        return image

    def detect_emotions(self):
        cap = cv2.VideoCapture(1)
        with self.__mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.detections:
                    for detection in results.detections:
                        face_coordinates, gray_face = self.__facepad(image, detection)
                        emotion, score = self.predict_emotion(gray_face)
                        logging.debug('{0}: ({1} {2})'.format(time.time(), emotion, score))
                        image = self.display_results(image, face_coordinates, emotion, score)
                    cv2.imshow('MediaPipe Face Detection', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        cap.release()


FER().detect_emotions()