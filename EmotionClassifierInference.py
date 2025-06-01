import cv2
import numpy as np
import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
# import os
# os.environ["TFLITE_ENABLE_XNNPACK"] = "0"
# from tensorflow.lite.python.interpreter import Interpreter

class EmotionClassifierInference:
    def __init__(self):
        self.emotion_label = ['happy', 'neutral', 'negative']
        super().__init__()

    def classifier_float(self, input_img, input_size):
        model = self.__create_model_efficient_lite0(input_size=128, num_classes=3)
        model.load_weights("./models/model_float_weights.h5")
        input_img = self.preproc(input_img)
        input_img = self.__make_input(input_img, input_size)
        pred = model.predict(input_img)
        e_label = self.emotion_label[np.argmax(pred)]
        return e_label

    # def classifier(self, input_img, input_size):
    #     # TFLiteモデルをロード（TensorFlow 本体を使用）
    #     interpreter = Interpreter(
    #         model_path="./models/fer2013_qat_model.tflite",
    #         experimental_delegates=[]  # 念のため
    #     )
    #     interpreter.allocate_tensors()
    #     print("allocate")

    #     # 入出力の情報を取得
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     input_index = input_details[0]['index']
    #     output_index = output_details[0]['index']

    #     # Debug出力
    #     print(f"Expected shape: {input_details[0]['shape']}")
    #     print(f"Expected dtype: {input_details[0]['dtype']}")

    #     # 前処理
    #     input_img = self.preproc(input_img)
    #     input_img = self.__make_input(input_img, input_size)
    #     print(f"Input shape: {input_img.shape}")
    #     print(f"Input dtype: {input_img.dtype}")

    #     # 型チェックと変換（必要なら）
    #     # チェックと修正（保険）
    #     if input_img.shape != tuple(input_details[0]['shape']):
    #         print(f"Fixing shape: input_img.shape={input_img.shape} → expected={input_details[0]['shape']}")
    #         input_img = np.reshape(input_img, input_details[0]['shape'])

    #     if input_img.dtype != input_details[0]['dtype']:
    #         print(f"Fixing dtype: input_img.dtype={input_img.dtype} → expected={input_details[0]['dtype']}")
    #         input_img = input_img.astype(input_details[0]['dtype'])
            
    #     # 推論を実行
    #     interpreter.set_tensor(input_index, input_img)
    #     interpreter.invoke()
    #     output = interpreter.get_tensor(output_index)
    #     print("Output shape:", output.shape)
    #     print("Output dtype:", output.dtype)

    #     # 結果の表示
    #     e_label = self.emotion_label[np.argmax(output[0])]
    #     print(e_label)
    #     return e_label

    def preproc(self, img, iterations=1, kernel_size=2):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール化
        # processed = img_gray.copy()
        # processed = cv2.equalizeHist(processed) # 明暗のバランス調整
        # # ぼかし画像を作成
        # blur = cv2.GaussianBlur(processed, (3, 3), sigmaX=3)
        # # シャープ画像 = 元画像 + α(元画像 - ぼかし画像)
        # processed = cv2.addWeighted(processed, 1.5, blur, -0.5, 0)
        # # processed = cv2.GaussianBlur(processed, (5, 5), 0)  # ぼかし
        # # kernel = np.ones((kernel_size, kernel_size), np.uint8) # カーネル作成
        # # for _ in range(iterations):
        # #     # オープニング：縮小→膨張（小さいノイズ除去）
        # #     processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        # #     # クロージング：膨張→縮小（小さい穴を埋める）
        # #     processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        # comps = cv2.hconcat([img_gray, processed])
        # cv2.imwrite("./test/output/input.png", comps)
        return img_gray

    def __make_input(self, img, size):
        face = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)  # リサイズ(48, 48)
        face = np.expand_dims(face, axis=-1)                              # (48,48) → (48,48,1)
        face = np.expand_dims(face, axis=0)                               # (48,48,1) → (1,48,48,1)
        face = face.astype(np.uint8)                                      # データ型をuint8に変換
        return face

    def __create_model_efficient_lite0(self, input_size=128, num_classes=3):
        inp = tf.keras.layers.Input(shape=(input_size, input_size, 1)) # 入力：128×128×3
        base = tf.keras.applications.EfficientNetB0( # EfficientNetB0 を Lite0 相当に軽量化
            input_tensor=inp,
            include_top=False,
            weights=None,       # ImageNet 事前学習不要 or 利用不可なら None
            pooling='avg'
        )
        x = base.output
        x = tf.keras.layers.Dropout(0.2)(x) # Dropout で過学習防止
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # 出力層
        model = tf.keras.Model(inputs=inp, outputs=out, name='EfficientLite0_128')
        return model

if __name__ == "__main__":
    ec = EmotionClassifierInference()
    # test
    print("================happy=================")
    happy = cv2.imread("./test/output/happy_face10_5.png")
    happy = cv2.resize(happy, (128, 128))
    ec.classifier(happy)
    print("================angry=================")
    angry = cv2.imread("./test/output/angry_face10_5.png")
    angry = cv2.resize(angry, (128, 128))
    ec.classifier(angry)
    print("================ sad =================")
    sad = cv2.imread("./test/output/sad_face10_5.png")
    sad = cv2.resize(sad, (128, 128))
    ec.classifier(sad)
