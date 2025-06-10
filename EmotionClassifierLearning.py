import cv2
import os 
import logging
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd
from pathlib import Path
from imgaug import augmenters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils.LoggingCallback import LoggingCallback
# from EmotionClassifierInference import EmotionClassifierInference

class EmotionClassifierLearning:
    def __init__(self):
        self.emotion_label = ['happy', 'neutral', 'negative']
        self.__log_config()
        super().__init__()

    def model_learning_efficient_lite0(self, dpath, input_size):
        X, Y = self.__data_loader(Path(dpath), input_size)
        X, Y = self.__data_augmentation(X, Y)
        Y = to_categorical(Y, num_classes=3)
        self.logger.info("==== Full-training model ====")
        model = self.__create_model_efficient_lite0(input_size=input_size, num_classes=3)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(X.astype(np.float32) / 255.0, Y, batch_size=16, epochs=50, validation_split=0.1, callbacks=self.__callbacks())
        model.save_weights("models/model_float_weights.h5")

    def __data_loader(self, dpath, input_size):
        data = pd.read_csv(dpath / "fer2013.csv")  # kaggleから取得したCSV形式を想定
        pixels = data["pixels"].tolist()
        X = np.array([cv2.resize(np.fromstring(p, sep=' ').reshape(48, 48, 1), (input_size, input_size)) for p in pixels])
        X = X.astype('uint8')
        # 変換マップ（7→3クラス）
        conversion_map = {
            0: 2,  # Angry → Negative
            1: 2,  # Disgust → Negative
            2: 2,  # Fear → Negative
            3: 0,  # Happy → Happy
            4: 2,  # Sad → Negative
            5: 1,  # Surprise → Neutral
            6: 1   # Neutral → Neutral
        }
        Y_mapped = np.vectorize(conversion_map.get)(data["emotion"]) # remap
        # log出力
        values, counts = np.unique(Y_mapped, return_counts=True)
        self.logger.info(f"Train-Dataset->{self.emotion_label[values[0]]}:{counts[0]}, {self.emotion_label[values[1]]}:{counts[1]}, {self.emotion_label[values[2]]}:{counts[2]}")
        return X, Y_mapped

    def __data_augmentation(self, X, Y):
        # Augmentationパイプライン
        augmentor = augmenters.Sequential([
            augmenters.Fliplr(0.5),             # 左右反転
            augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}), # 平行移動
            augmenters.Add((-20, 20)),          # 輝度の変化
            augmenters.Multiply((0.8, 1.2))     # コントラスト
        ])
        # 0:Happyと1:Neutralだけを選択
        indices_to_augment_0 = np.where(np.isin(Y, [0]))[0]
        indices_to_augment_1 = np.where(np.isin(Y, [1]))[0]
        aug_0 = len(Y) - 2 * len(indices_to_augment_0) - len(indices_to_augment_1)
        aug_1 = len(Y) - len(indices_to_augment_0) - 2 * len(indices_to_augment_1)
        # データを抽出: 最大サンプル数に合わせてデータ拡張
        indices_to_augment_0_smp = np.random.choice(indices_to_augment_0, size=aug_0, replace=False)
        indices_to_augment_1_smp = np.random.choice(indices_to_augment_1, size=aug_1, replace=False)
        # augmentation
        X_aug_target_0 = X[indices_to_augment_0_smp]
        Y_aug_target_0 = Y[indices_to_augment_0_smp]
        X_augmented_0 = augmentor(images=X_aug_target_0)
        X_aug_target_1 = X[indices_to_augment_1_smp]
        Y_aug_target_1 = Y[indices_to_augment_1_smp]
        X_augmented_1 = augmentor(images=X_aug_target_1)
        X_augmented = np.concatenate([X_augmented_0, X_augmented_1], axis=0)
        Y_aug_target = np.concatenate([Y_aug_target_0, Y_aug_target_1], axis=0) 
        # 元データと結合
        X_combined = np.concatenate([X, X_augmented], axis=0)
        Y_combined = np.concatenate([Y, Y_aug_target], axis=0)
        # log出力
        values, counts = np.unique(Y_combined, return_counts=True)
        self.logger.info(f"Train-Dataset(after augmentationed)->{self.emotion_label[values[0]]}:{counts[0]}, {self.emotion_label[values[1]]}:{counts[1]}, {self.emotion_label[values[2]]}:{counts[2]}")
        return X_combined, Y_combined

    # def __create_model(self):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    #         tf.keras.layers.MaxPooling2D((2,2)),
    #         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #         tf.keras.layers.MaxPooling2D((2,2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(3, activation='softmax') # クラス縮約に伴って3に変更
    #     ])
    #     return model
    
    def __create_model_efficient_lite0(self, input_size=128, num_classes=3):
        inp = tf.keras.layers.Input(shape=(input_size, input_size, 1)) # 入力：128×128×3
        inp_3 = tf.keras.layers.Concatenate()([inp, inp, inp])
        base = tf.keras.applications.EfficientNetB0( # EfficientNetB0 を Lite0 相当に軽量化
            input_tensor=inp_3,
            include_top=False,
            weights=None,       # ImageNet 事前学習不要 or 利用不可なら None
            pooling='avg'
        )
        x = base.output
        x = tf.keras.layers.Dropout(0.2)(x) # Dropout で過学習防止
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # 出力層
        model = tf.keras.Model(inputs=inp, outputs=out, name='EfficientLite0')
        return model

    def __log_config(self):
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("./models/EC_model_learning.log", mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def __callbacks(self):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',    # 検証用ロスを監視
            min_delta=1e-3,            # 「改善」と見なす最小の変化
            patience=15,               # 改善が見られないエポック数
            verbose=1,                 # ログ出力あり
            mode='auto',               # 自動で最小化 or 最大化を判定
            baseline=None,             # 指標の初期基準値（指定しない）
            restore_best_weights=True  # 学習終了時に最良の重みに戻す
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',    # 検証用ロスを監視
            factor=0.5,                # 学習率を半分にする
            patience=7,                # 改善が見られないエポック数
            verbose=1,                 # ログ出力あり
            mode='auto',               # 自動判定（'min' or 'max'）
            min_delta=1e-3,            # 「改善」と見なす最小の差分
            cooldown=2,                # 学習率変更後、待機するエポック数
            min_lr=1e-6                # 最小の学習率（これ以下には下げない）
        )
        tensorboard = TensorBoard(log_dir='./TensorBoard', histogram_freq=1)
        logging_callback = LoggingCallback(self.logger)
        return [early_stopping, reduce_lr, tensorboard, logging_callback]

if __name__ == "__main__":
    ec = EmotionClassifierLearning()
    ec.model_learning_efficient_lite0(dpath="./dataset", input_size=192)