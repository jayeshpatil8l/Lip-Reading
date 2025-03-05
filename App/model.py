import os
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Input, TimeDistributed, Flatten, Attention # type: ignore
from tensorflow.keras.models import Model # type: ignore

def build_model(input_shape, vocab_size):

    input = Input(shape = input_shape)

    # Convolutional layers
    x = Conv3D(128, 3, padding='same')(input)
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
    lstm_output = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    lstm_output = Dropout(0.5)(lstm_output)

    lstm_output = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(lstm_output)
    lstm_output = Dropout(0.5)(lstm_output)

    # Attention layer
    attention_output = Attention(use_scale=True, score_mode='dot')([lstm_output, lstm_output])

    # Dense output layer
    output_layer = Dense(vocab_size + 1, kernel_initializer='he_normal', activation='softmax')(attention_output)

    # Build the model
    model = Model(inputs = input, outputs = output_layer)

    model.load_weights(filepath = os.path.join(".","models","model_with_attention_weights_after_50.weights.h5"))

    # Print the model summary
    return model
