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
        # kernel = np.ones((kernel_size, kernel_size), np.uint8) # カーネル作成
        # processed = cv2.GaussianBlur(processed, (5, 5), 0)  # ぼかし
        # for _ in range(iterations):
        #     # オープニング：縮小→膨張（小さいノイズ除去）
        #     processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        #     # クロージング：膨張→縮小（小さい穴を埋める）
        #     processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
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
    """
    全体動作を考えると、
    ・入力はグレースケール
    ・出力は検出した顔部分をクリップし、(48, 48)にリサイズした画像
    """
    fd = FaceDetector()
    iterations = 10
    kernel_size = 5
    print("================happy=================")
    happy = cv2.resize(cv2.imread("./test/happy.png"), (600, 400), interpolation=cv2.INTER_LINEAR) # 入力+リサイズ
    happy_gray = cv2.cvtColor(happy, cv2.COLOR_BGR2GRAY)
    happy_processed, happy_face = fd.face_detection(happy_gray, iterations=iterations, kernel_size=kernel_size)
    happy_face = cv2.resize(happy_face, (48, 48), interpolation=cv2.INTER_LINEAR)
    comps = cv2.hconcat([happy_gray, happy_processed])
    cv2.imwrite("./test/output/happy_"+str(iterations)+"_"+str(kernel_size)+".png", comps)
    cv2.imwrite("./test/output/happy_face"+str(iterations)+"_"+str(kernel_size)+".png", happy_face)
    print("================angry=================")
    angry = cv2.resize(cv2.imread("./test/angry.png"), (600, 400), interpolation=cv2.INTER_LINEAR) # 入力+リサイズ
    angry_gray = cv2.cvtColor(angry, cv2.COLOR_BGR2GRAY)
    angry_processed, angry_face = fd.face_detection(angry_gray, iterations=iterations, kernel_size=kernel_size)
    angry_face = cv2.resize(angry_face, (48, 48), interpolation=cv2.INTER_LINEAR)
    comps = cv2.hconcat([angry_gray, angry_processed])
    cv2.imwrite("./test/output/angry_"+str(iterations)+"_"+str(kernel_size)+".png", comps)
    cv2.imwrite("./test/output/angry_face"+str(iterations)+"_"+str(kernel_size)+".png", angry_face)
    print("================ sad =================")
    sad = cv2.resize(cv2.imread("./test/sad.png"), (600, 400), interpolation=cv2.INTER_LINEAR) # 入力+リサイズ
    sad_gray = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)
    sad_processed, sad_face = fd.face_detection(sad_gray, iterations=iterations, kernel_size=kernel_size)
    sad_face = cv2.resize(sad_face, (48, 48), interpolation=cv2.INTER_LINEAR)
    comps = cv2.hconcat([sad_gray, sad_processed])
    cv2.imwrite("./test/output/sad_"+str(iterations)+"_"+str(kernel_size)+".png", comps)
    cv2.imwrite("./test/output/sad_face"+str(iterations)+"_"+str(kernel_size)+".png", sad_face)