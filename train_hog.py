import argparse
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import glob
import numpy as np
import pandas as pd
import json
import shutil
import tensorflow as tf
import cv2
from utils import plot
import pickle
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Lambda, Activation, GRU, Dropout, Flatten, Masking
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping
from data_generator import Generator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

idx2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
              11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
              22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'NULL'}
VOCAB = len(idx2letter)


def create_model():
    encoder_input = Input(name="encoder_input", shape=(None, 128))  # n feature vectors 512-dim each
    masked_input = Masking(mask_value=100.0, name='masked_input')(encoder_input)
    labels = Input(name='the_labels', shape=[13], dtype="int32")  # shape is max word length

    output = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(masked_input)
    output = Dropout(0.5)(output)
    output = Bidirectional(GRU(units=64, return_sequences=True, dropout=0.4))(output)
    output = Bidirectional(GRU(units=32, return_sequences=True, dropout=0.4))(output)
    y_pred = Dense(VOCAB + 1, name="dense", activation="softmax")(output)

    loss_out = CTCLayer(name="ctc_loss")(labels, y_pred)

    model = Model([encoder_input, labels], loss_out)
    # Define the model

    sgd = SGD(lr=1e-3, decay=1e-2)  # decay 1e-6 resulted to 30 loss

    model.compile(optimizer=sgd)
    model.summary()

    return model


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int32")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int32")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int32")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss((loss))

        # At test time, just return the computed predictions
        return y_pred


def train(user):
    total_time = 0
    num_epochs = 20
    batch_size = 8
    model = create_model()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    start_time = time.time()
    if user == "all":
        print("Training for all users")
        total_train = True
    else:
        print("Training for user {}".format(user))
        total_train = False
    train_generator = Generator(os.path.join("hog", user), "train", batch_size, user, total_train)
    val_generator = Generator(os.path.join("hog", user), "dev", batch_size, user, total_train)

    hist = model.fit(train_generator,
                     validation_data=val_generator,
                     epochs=num_epochs,
                     callbacks=[early_stopping], shuffle=False)
    end_time = time.time()
    model.save("models/hog/hog_model")
    model.save_weights("models/hog/hog_model")
    print("Saved model!")
    plot(hist, num_epochs)

    train_time = end_time - start_time
    total_time += train_time
    print("Total training time: {}".format(total_time))

    with open("models/hog/hog_model/history", "wb") as file:
        pickle.dump(hist.history, file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin', 'all'],
                    help='choose user')
    args = ap.parse_args()
    user = args.users

    train(user)
