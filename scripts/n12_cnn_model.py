import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class CNNClassifierModel:
    def __init__(self, input_dim: int, class_num: int, best_model_path: str):
        """
        input_dim       — размер входного вектора
        class_num       — количество классов
        best_model_path — путь для сохранения лучшей модели
        """
        self.input_dim = input_dim
        self.class_num = class_num
        self.best_model_path = best_model_path
        self.model = self._build_model()

    # ---------------------------
    # MODEL DEFINITION
    # ---------------------------
    def _build_model(self):
        model = Sequential()

        model.add(Input(shape=(self.input_dim, 1)))

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

        model.add(Dropout(0.5))

        # flatten + dense(128)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))

        model.add(Dense(self.class_num, activation=None))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy'
            ]
        )

        return model

    # ---------------------------
    # TRAIN FUNCTION
    # ---------------------------
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=20,
        batch_size=32
    ):
        callbacks = []

        X_train = np.expand_dims(X_train, axis=-1)

        if X_val is not None and y_val is not None:
            X_val = np.expand_dims(X_val, axis=-1)

            checkpoint = ModelCheckpoint(
                filepath=self.best_model_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        if X_val is not None and y_val is not None:
            self.model = tf.keras.models.load_model(self.best_model_path)

        return history

    def predict(self, X):
        X = np.expand_dims(X, axis=-1)

        logits = self.model.predict(X)
        classes = np.argmax(logits, axis=1)

        return classes, logits

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)