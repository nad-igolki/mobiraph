import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class CNNClassifierModel:
    def __init__(self, input_dim: int, class_num: int):
        """
        input_dim  — размер входного вектора
        class_num  — количество классов
        """
        self.input_dim = input_dim
        self.class_num = class_num
        self.model = self._build_model()

    # ---------------------------
    # MODEL DEFINITION
    # ---------------------------
    def _build_model(self):

        model = Sequential()

        model.add(Dense(128,
                        activation='relu',
                        input_shape=(self.input_dim,)))

        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(self.class_num,
                        activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
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

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val)
            if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return history

    # ---------------------------
    # INFERENCE FUNCTION
    # ---------------------------
    def predict(self, X):

        probs = self.model.predict(X)
        classes = np.argmax(probs, axis=1)

        return classes, probs

    # ---------------------------
    # SAVE / LOAD
    # ---------------------------
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)