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
        input_img3 = tf.keras.layers.Concatenate()([input_img, input_img, input_img])
        pred = model.predict(input_img3)
        e_label = self.emotion_label[np.argmax(pred)]
        return e_label

    def preproc(self, img, iterations=1, kernel_size=2):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # グレースケール化
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
