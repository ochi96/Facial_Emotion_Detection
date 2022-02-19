import cv2
import numpy as np
import time
from mtcnn.mtcnn import MTCNN

from typing import Sequence, Tuple, Union

NumpyRects = Union[np.ndarray, Sequence[Tuple[int, int, int, int]]]
PADDING = 40


class FER():

    def __init__(self, mtcnn=False, emotion_model: str = None, scale_factor: float = 1.1,
                        min_face_size: int = 50, min_neighbors: int = 5, offsets: tuple = (10, 10), compile: bool = False) -> None:

        self.__scale_factor = scale_factor
        self.__min_neighbors = min_neighbors
        self.__min_face_size = min_face_size
        self.__offsets = offsets

        cascade_file = 'Models/haarcascade_frontalface_default.xml'
        if mtcnn:
            self.__face_detector = "mtcnn"
            self._mtcnn = MTCNN()
        else:
            self.__face_detector = cv2.CascadeClassifier(cascade_file)
        
        # Local Keras model

        self.__emotion_classifier = tf.lite.Interpreter(base_dir + 'Models/production_fer_model.tflite')
        self.__emotion_classifier.allocate_tensors()
        
        self.__emotion_target_size = (64,64)
        # print

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

    def find_faces(self, img: np.ndarray, bgr=True) -> list:
        """Image to list of faces bounding boxes(x,y,w,h)"""
        if isinstance(self.__face_detector, cv2.CascadeClassifier):
            if bgr:
                gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # assume gray
                gray_image_array = img

        faces = self.__face_detector.detectMultiScale(
            gray_image_array,
            scaleFactor=self.__scale_factor,
            minNeighbors=self.__min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(self.__min_face_size, self.__min_face_size),
        )
        elif self.__face_detector == "mtcnn":
            results = self._mtcnn.detect_faces(img)
            faces = [x["box"] for x in results]
        return faces

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

        if (face_rectangles is None):
            face_rectangles = self.find_faces(img, bgr=True)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = self.pad(gray_img)

        emotions = []
        for face_coordinates in face_rectangles:
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

        try:
            gray_face = cv2.resize(gray_face, self.__emotion_target_size)
        except Exception as e:
            continue
        
        gray_face = self.__preprocess_input(gray_face, True)
        gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)

        start = time.time()

        self.__emotion_classifier.set_tensor(0, gray_face)
        self.__emotion_classifier.invoke()
        emotion_prediction = self.__emotion_classifier.get_tensor(77)[0]

        # emotion_prediction = self.__emotion_classifier.predict(gray_face)[0]
        labelled_emotions = {
            emotion_labels[idx]: round(float(score), 2)
            for idx, score in enumerate(emotion_prediction)
        }

        emotions.append(
            dict(box=face_coordinates, emotions=labelled_emotions)
        )

        self.emotions = emotions
        end = time.time()
        print('Inference only: ', end-start)
        return emotions

    def top_emotion(self, img: np.ndarray)-> Tuple[Union[str,None], Union[float,None]]:
        """Convenience wrapper for `detect_emotions` returning only top emotion for first face in frame.
        :param img: image to process
        :return: top emotion and score (for first face in frame) or (None, None)
        """
        emotions = self.detect_emotions(img=img)
        print(emotions)

        # rects = emotions[0]['box']

        # label_position = (rects[i][0] + int((rects[i][1] / 3)), abs(rects[i][2] - 10))

        top_emotions = [max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions]

        # Take first face
        if len(top_emotions):
            top_emotion = top_emotions[0]
            print(top_emotion)
            # rects = top_emotion['box']
            # print(rects)
        else:
            return (None, None)
        
        score = emotions[0]["emotions"][top_emotion]
        # print('Inference only: ', end-start)

        return emotions, top_emotion, score

  

def text_on_detected_boxes(text, emotions,image,font_scale = 1,
                           font = cv2.FONT_HERSHEY_SIMPLEX,
                           FONT_COLOR = (0, 0, 0),
                           FONT_THICKNESS = 2,
                           rectangle_bgr = (0, 255, 0)):
  
    rects = emotions[0]['box']
    x, y, w, h = rects
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


    text_x, text_y = (x*2 , y-10)

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # Set the Coordinates of the boxes
    box_coords = ((text_x-10, text_y+4), (text_x + text_width+10, text_y - text_height-5))
    # Draw the detected boxes and labels
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, fontScale=font_scale, color=FONT_COLOR,thickness=FONT_THICKNESS)
    cv2.imwrite(base_dir + "Data/Test/new_gen.jpg", image)
    pass

image = cv2.imread(base_dir +"Data/Test/justin.jpg")
start = time.time()
fer = FER(mtcnn=True)
emotions, top_emotion, score = fer.top_emotion(image)
text ='{}{}'.format(top_emotion, score)
print(text)
text_on_detected_boxes(text, emotions, image)


end = time.time()
print(top_emotion, score)
print('Total Time: ', end-start )
