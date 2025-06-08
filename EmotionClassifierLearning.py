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
    
    # def model_learning(self, dpath):
    #     X, Y = self.__data_loader(Path(dpath), 48)
    #     X, Y = self.__data_augmentation(X, Y)
    #     Y = to_categorical(Y, num_classes=3)
    #     # floatモデルで事前学習
    #     self.logger.info("==== Full-training model ====")
    #     model_fp32 = self.__create_model()
    #     model_fp32.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     model_fp32.fit(X.astype('float32') / 255.0, Y, batch_size=8, epochs=1, validation_split=0.1, callbacks=self.__callbacks())
    #     model_fp32.save_weights("models" / "model_float_weights.h5")
    #     # QATモデル変換
    #     self.logger.info("==== QAT-training model ====")
    #     quantize_model = tfmot.quantization.keras.quantize_model
    #     model_qat = quantize_model(model_fp32)
    #     model_qat.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    #     # QAT学習
    #     model_qat.fit(X, Y, batch_size=8, epochs=1, validation_split=0.1, callbacks=self.__callbacks())
    #     # TFLite uint8 変換
    #     converter = tf.lite.TFLiteConverter.from_keras_model(model_qat)
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    #     converter.inference_input_type = tf.uint8  # 実行時の入力型
    #     converter.inference_output_type = tf.uint8
    #     converter.experimental_new_converter = False #True

    #     def representative_dataset():
    #         for data in tf.data.Dataset.from_tensor_slices(X[:100]).batch(1):
    #             yield [tf.cast(data, tf.float32)]

    #     converter.representative_dataset = representative_dataset
    #     tflite_model = converter.convert()
    #     with open("./models/fer2013_qat_model.tflite", "wb") as f:
    #         f.write(tflite_model)
    #     self.logger.info("Trained quantized model saved to fer2013_qat_model.tflite")

    def model_learning_efficient_lite0(self, dpath, input_size):
        X, Y = self.__data_loader(Path(dpath), input_size)
        X, Y = self.__data_augmentation(X, Y)
        Y = to_categorical(Y, num_classes=3)
        # floatモデルで事前学習
        self.logger.info("==== Full-training model ====")
        model = self.__create_model_efficient_lite0(input_size=input_size, num_classes=3)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(X.astype(np.float32) / 255.0, Y, batch_size=16, epochs=50, validation_split=0.1, callbacks=self.__callbacks())
        model.save_weights("models/model_float_weights.h5")
        # QAT 変換・ファインチューニング
        self.logger.info("==== QAT-training model ====")
        self.quantize(input_size)
        # model = self.__create_model_efficient_lite0(input_size=input_size, num_classes=3)
        # model.load_weights("models/model_float_weights.h5")
        # # qat_model = tfmot.quantization.keras.quantize_model(model)
        # annotated_model = tfmot.quantization.keras.quantize_annotate_model(model) # Rescaling を量子化せず、残りをアノテーションする
        # qat_model = tfmot.quantization.keras.quantize_apply(annotated_model) # 量子化モデルに変換
        # qat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
        #                 loss='categorical_crossentropy',
        #                 metrics=['accuracy'])
        # qat_model.fit(X, Y, batch_size=8, epochs=1, validation_split=0.1, callbacks=self.__callbacks())

        # # TFLite int8 量子化変換
        # converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        # converter.inference_input_type  = tf.uint8
        # converter.inference_output_type = tf.uint8

        # def representative_dataset():
        #     for data in tf.data.Dataset.from_tensor_slices(X[:100]).batch(1):
        #         yield [tf.cast(data, tf.float32)]

        # converter.representative_dataset = representative_dataset

        # tflite_model = converter.convert()
        # with open('efficient_lite0_128_int8.tflite','wb') as f:
        #     f.write(tflite_model)

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
        # Augmentationパイプライン（例）
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

    def __create_qat_model_efficient_lite0(self, input_size=128, num_classes=3):
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(128, 128, 3),
            include_top=False,
            weights=None,
            pooling='avg'
        )
        qat_base = tfmot.quantization.keras.quantize_annotate_model(base_model)

        inp = tf.keras.layers.Input(shape=(input_size, input_size, 1)) # 入力：128×128×3
        inp_3 = tf.keras.layers.Concatenate()([inp, inp, inp])
        x = qat_base(inp_3)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # アノテート済みモデルを丸ごと量子化
        annotated = tfmot.quantization.keras.quantize_annotate_model(model)
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_apply(annotated)
        return qat_model

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
        # checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
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

    def quantize(self, input_size):
        model = self.__create_model_efficient_lite0(input_size=input_size, num_classes=3)
        model.load_weights("models/model_float_weights.h5")
        print(model.summary())
        # Rescaling を除いてアノテート
        print("annoated...")
        # annotated_model = self.annotate_model_excluding_rescaling(model)
        annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
        # QAT モデルに変換
        print("quantize...")
        # qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        with tfmot.quantization.keras.quantize_scope():
            model_qat = tfmot.quantization.keras.quantize_apply(annotated_model)
        # annotated_model = tfmot.quantization.keras.quantize_annotate_model(model) # Rescaling を量子化せず、残りをアノテーションする
        # qat_model = tfmot.quantization.keras.quantize_apply(annotated_model) # 量子化モデルに変換
        print("compile...")
        qat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        print("start learning...")
        qat_model.fit(X, Y, batch_size=8, epochs=1, validation_split=0.1, callbacks=self.__callbacks())

        # TFLite int8 量子化変換
        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8

        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices(X[:100]).batch(1):
                yield [tf.cast(data, tf.float32)]

        converter.representative_dataset = representative_dataset

        tflite_model = converter.convert()
        with open('efficient_lite0_quantize.tflite','wb') as f:
            f.write(tflite_model)

    def annotate_model_excluding_rescaling(self, model):
        annotated_layers_map = {}

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                annotated_layers_map[layer.name] = layer
            else:
                annotated_layers_map[layer.name] = tfmot.quantization.keras.quantize_annotate_layer(layer)

        # 入力から順にアノテート済みレイヤーを再構成する
        inputs = model.input
        x = inputs
        for layer in model.layers[1:]:  # 入力層はもうある
            annotated_layer = annotated_layers_map[layer.name]
            if isinstance(x, list):
                x = annotated_layer(x)
            else:
                x = annotated_layer([x]) if isinstance(layer.input, list) else annotated_layer(x)

        return Model(inputs, x)

if __name__ == "__main__":
    ec = EmotionClassifierLearning()
    ec.model_learning_efficient_lite0(dpath="./dataset", input_size=192)
    # ec.quantize(192)
