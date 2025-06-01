import cv2
import numpy as np
from EmotionClassifierInference import EmotionClassifierInference
from FaceDetector import FaceDetector
from JukeBox import JukeBox

"""
This tool plays cheerful music when a happy facial expression is detected.
[System Flow: camera -> FaceDetector -> EmotionClassifier -> JukeBox]

camera: web-cam(HD 1080P)
FaceDetector: Detect the face and crop the facial region to the specified input size.
EmotionClassifier: Recognize the facial expression of the input face.
JukeBox: Play music.
"""

class InterfaceContoroller(FaceDetector, EmotionClassifierInference, JukeBox):
    def __init__(self):
        super().__init__()

    def capture(self, input_size=48, emotion_th=10, model_type="quantize"):
        assert model_type == "quantize" or model_type == "float"
        emo_cnt = 0
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, face = self.face_detection(img, input_size) # If no face is detected, a completely black image is returned.
            label = self.classifier(face, input_size) if model_type=="quantize" else self.classifier_float(face, input_size)
            # If the target expression is detected continuously, the music will start playing.
            if label == "happy":
                emo_cnt += 1
            else:
                emo_cnt = 0
            if emo_cnt >= emotion_th:
                emo_cnt = 0
                flag = self.control_music()
                if flag:
                    break
            print(label, emo_cnt)
            
        cap.release()
        cv2.destroyAllWindows()

    def control_music(self):
        self.play_with_interrupt()
        flag = 0 if input("Continue? -> yes/no: ") == "yes" else 1
        return flag

if __name__ == "__main__":
    ic = InterfaceContoroller()
    ic.capture(input_size=128, emotion_th=1, model_type="float")