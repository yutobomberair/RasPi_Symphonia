import cv2
import numpy as np

class FaceDetector():
    def __init__(self):
        #分類器を読み込む
        CASCADE_FILE= "models/haarcascade_frontalface_default.xml"
        self.CLAS = cv2.CascadeClassifier(CASCADE_FILE)
        super().__init__()

    def preproc(self, img, iterations=5, kernel_size=3):
        processed = img.copy()
        processed = cv2.equalizeHist(processed) # 明暗のバランス調整
        return processed

    def face_detection(self, img, size=128, scaleFactor=1.1, minNeighbors=5, flags=0, minSize=(0, 0), iterations=10, kernel_size=3):
        img_processed = self.preproc(img, iterations, kernel_size)
        face_key = self.CLAS.detectMultiScale(img_processed, scaleFactor = scaleFactor, minNeighbors=minNeighbors, flags=flags, minSize = minSize)
        if len(face_key) == 0:
            face_img = np.zeros((size, size), dtype=np.uint8)
        else:
            face_img = self.__clip_coor(img, face_key) # 切り取る画像はオリジナル
        return img_processed, face_img

    def __clip_coor(self, img, face_key, offset=50):
        width, height = img.shape[1], img.shape[0]
        x, y, w, h = face_key[0][0], face_key[0][1], face_key[0][2], face_key[0][3]
        if x > offset:
            x -= offset
        if y > offset:
            y -= offset
        if x + w + offset < width:
            w += offset
        if y + h + offset < height:
            h += offset
        print()
        return img[y:y+h, x:x+w]
    
if __name__ == "__main__":
    fd = FaceDetector()