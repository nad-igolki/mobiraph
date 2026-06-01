import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# =========================
# FEATURE TOKENIZER
# =========================
class NumericalFeatureTokenizer(layers.Layer):
    """
    Превращает каждый числовой признак в embedding размерности d_token.
    Вход:  (batch, n_features)
    Выход: (batch, n_features, d_token)
    """
    def __init__(self, n_features: int, d_token: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_token = d_token

    def build(self, input_shape):
        # Для каждого признака: x_i * w_i + b_i, где w_i и b_i имеют размер d_token
        self.weight = self.add_weight(
            name="weight",
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_features, self.d_token),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        # x: (batch, n_features)
        x = tf.expand_dims(x, axis=-1)                  # (batch, n_features, 1)
        tokens = x * self.weight + self.bias            # (batch, n_features, d_token)
        return tokens


# =========================
# TRANSFORMER BLOCK
# =========================
class TransformerBlock(layers.Layer):
    def __init__(self, d_token: int, num_heads: int, ff_dim: int, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_token // num_heads,
            dropout=dropout
        )
        self.dropout1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_token),
        ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention + residual
        attn_input = self.norm1(x)
        attn_output = self.attn(attn_input, attn_input, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        # FFN + residual
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output

        return x


# =========================
# FT-TRANSFORMER MODEL
# =========================
class FTTransformerClassifierModel:
    def __init__(
        self,
        input_dim: int,
        class_num: int,
        best_model_path: str,
        d_token: int = 64,
        num_heads: int = 8,
        ff_dim: int = 128,
        num_transformer_blocks: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.class_num = class_num
        self.best_model_path = best_model_path

        self.d_token = d_token
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.input_dim,), dtype=tf.float32)

        # 1) Tokenize numerical features
        x = NumericalFeatureTokenizer(
            n_features=self.input_dim,
            d_token=self.d_token
        )(inputs)  # (batch, input_dim, d_token)

        # 2) CLS token
        cls_token = tf.Variable(
            initial_value=tf.random.normal((1, 1, self.d_token)),
            trainable=True,
            name="cls_token",
            dtype=tf.float32,
        )

        def add_cls_token(batch_tokens):
            batch_size = tf.shape(batch_tokens)[0]
            cls = tf.repeat(cls_token, repeats=batch_size, axis=0)  # (batch, 1, d_token)
            return tf.concat([cls, batch_tokens], axis=1)

        x = layers.Lambda(add_cls_token)(x)  # (batch, input_dim + 1, d_token)

        # 3) Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                d_token=self.d_token,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )(x)

        # 4) Take CLS representation
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        cls_repr = x[:, 0, :]   # (batch, d_token)

        # 5) Head
        cls_repr = layers.Dense(128, activation="relu")(cls_repr)
        cls_repr = layers.Dropout(self.dropout)(cls_repr)
        outputs = layers.Dense(self.class_num, activation=None)(cls_repr)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        return model

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=2,
        batch_size=32
    ):
        callbacks = []

        if X_val is not None and y_val is not None:
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
            self.model = tf.keras.models.load_model(
                self.best_model_path,
                custom_objects={
                    "NumericalFeatureTokenizer": NumericalFeatureTokenizer,
                    "TransformerBlock": TransformerBlock,
                }
            )

        return history

    def predict(self, X):
        logits = self.model.predict(X)
        classes = np.argmax(logits, axis=1)
        return classes, logits

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                "NumericalFeatureTokenizer": NumericalFeatureTokenizer,
                "TransformerBlock": TransformerBlock,
            }
        )