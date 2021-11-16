from tensorflow import keras
from tensorflow.keras import layers
import pickle

def get_biLSTM():
    inp = keras.Input(shape=(None,), dtype='int64')

    x = layers.Embedding(20002, 200)(inp)
    x = layers.SpatialDropout1D(0.4)(x)

    x_gru = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x_gru_dr = layers.Dropout(0.4)(x_gru)

    x1_conv1 = layers.Conv1D(128, 5, activation="relu")(x_gru_dr)
    x1_conv1_maxpool = layers.MaxPooling1D(5)(x1_conv1)
    x1_conv2 = layers.Conv1D(64, 5, activation="relu")(x1_conv1_maxpool)
    x1_avgpool = layers.GlobalAveragePooling1D()(x1_conv2)
    x1_maxpool = layers.GlobalMaxPool1D()(x1_conv2)

    x_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x_lstm_dr = layers.Dropout(0.4)(x_lstm)

    x2_conv1 = layers.Conv1D(128, 5, activation="relu")(x_lstm_dr)
    x2_conv1_maxpool = layers.MaxPooling1D(5)(x2_conv1)
    x2_conv2 = layers.Conv1D(64, 5, activation="relu")(x2_conv1_maxpool)
    x2_avgpool = layers.GlobalAveragePooling1D()(x2_conv2)
    x2_maxpool = layers.GlobalMaxPool1D()(x2_conv2)

    x = keras.layers.concatenate(
        [x1_avgpool, x1_maxpool, x2_avgpool, x2_maxpool]
    )

    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    preds = layers.Dense(1, activation='sigmoid')(x)

    model_cnn_biLSTM = keras.Model(inp, preds)

    return model_cnn_biLSTM


def get_e2e():
    biLSTM = get_biLSTM()
    biLSTM.load_weights('./weights/biLSTM')
    vectorizer = layers.TextVectorization(
        max_tokens=20000, output_sequence_length=150, standardize=None)
    vectorizer_weights = pickle.load(
        open("./weights/vectorizer.pkl", "rb"))['weights']
    vectorizer.set_weights(vectorizer_weights)

    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)
    out = biLSTM(x)
    e2e = keras.Model(inp, out)
    e2e.compile(loss="binary_crossentropy", optimizer='adam', metrics=["acc"])

    return e2e
