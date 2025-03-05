import os
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Input, TimeDistributed, Flatten, Attention # type: ignore
from tensorflow.keras.models import Model # type: ignore
import logging
import numpy as np
from utils import get_corrected_preds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LipReadingModel:
    def __init__(self):
        self.batch_size = 1
        self.height = 46
        self.width = 140
        self.channels = 1

        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary = self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )
        self.vocab_size = self.num_to_char.vocabulary_size()

        self.model = self.build_model()
        self.reset_model_states()


    def build_model(self):
        # Input layer
        input_layer = Input(batch_shape=(self.batch_size, None, self.height, self.width, self.channels))

        # Convolutional layers
        x = Conv3D(128, 3, padding='same')(input_layer)
        x = Activation('relu')(x)
        x = MaxPool3D((1, 2, 2))(x)

        x = Conv3D(256, 3, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool3D((1, 2, 2))(x)

        x = Conv3D(75, 3, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool3D((1, 2, 2))(x)

        # Flatten and TimeDistributed layer
        x = TimeDistributed(Flatten())(x)

        # Attention layer
        # attention_output = Attention(use_scale=True, score_mode='dot')([x, x])

        # Bidirectional LSTMs
        lstm_output = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', stateful = True, return_sequences=True))(x)
        lstm_output = Dropout(0.5)(lstm_output)

        lstm_output = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', stateful = True, return_sequences=True))(lstm_output)
        lstm_output = Dropout(0.5)(lstm_output)

        # Attention layer
        attention_output = Attention(use_scale=True, score_mode='dot')([lstm_output, lstm_output])

        # Dense output layer
        output_layer = Dense(self.vocab_size + 1, kernel_initializer='he_normal', activation='softmax')(attention_output)

        # Build the model
        model = Model(inputs=input_layer, outputs=output_layer)

        logger.info("Built the model architecture successfully.")

        model.load_weights(filepath = os.path.join(".","Models","model_f25.weights.h5"))

        logger.info("Model Loaded Successfully!")

        return model
    
    def predict(self, frames):
        frames_array = np.stack(frames, axis=0)  
        input_length = frames_array.shape[0]
        frames_array = np.expand_dims(frames_array, axis = 0)

        logits = self.model.predict(frames_array)
        decoded = tf.keras.backend.ctc_decode(logits, input_length=[input_length], greedy=True)[0][0].numpy()
        predicted = tf.strings.reduce_join([self.num_to_char(word) for word in decoded[0]]).numpy().decode('utf-8')
        prediction = get_corrected_preds(predicted)

        logger.info(f"Model Predicted: {prediction}")

        return prediction
    
    def reset_model_states(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Bidirectional):
                layer.reset_states()
        logger.info("Model States Reset Successfully!")