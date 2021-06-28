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
from tensorflow.keras.applications.vgg19 import VGG19
import cv2
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from utils import plot, return_classes
import pickle
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Lambda, Activation, GRU, Dropout, Flatten, Masking, \
    BatchNormalization, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_generator import Generator


idx2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
              11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y'
    , 25: 'z', 26: 'NULL'}

VOCAB = len(idx2letter)


def create_model():

    encoder_input = Input(name="encoder_input", shape=(None, 512))
    labels = Input(name='the_labels', shape=[13], dtype="int32")  # shape is max word length

    masked_input = Masking(mask_value=100.0, name='masked_input')(encoder_input)
    # masked_labels = Masking(mask_value=27, name='masked_labels')(labels)

    output = (LSTM(units=64, return_sequences=True, dropout=0.8))(masked_input)
    output = Dropout(0.3)(output)
    # output = LSTM(units=300, return_sequences=True, dropout=0.5)(output)
    # output = Dropout(0.3)(output)

    output = (LSTM(units=32, return_sequences=True, dropout=0.8))(output)
    output = Dropout(0.8)(output)
    output = BatchNormalization()(output)

    y_pred = Dense(VOCAB + 1, name="dense", activation="softmax")(output)

    loss_out = CTCLayer(name="ctc_loss")(labels, y_pred)

    model = Model([encoder_input, labels], loss_out)
    # Define the model
    sgd = SGD(lr=1e-2, decay=1e-2)
    model.compile(optimizer=sgd)

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
        self.add_loss(tf.reduce_mean(loss))

        # At test time, just return the computed predictions
        return y_pred


def train(user):
    total_time = 0
    num_epochs = 10
    batch_size = 32

    model = create_model()
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    start_time = time.time()

    if user == "all":
        print("Training for all users")
        total_train = True
    else:
        print("Training for user {}".format(user))
        total_train = False

    train_generator = Generator(os.path.join("resnet/data", user), "train", batch_size, user, total_train)
    val_generator = Generator(os.path.join("resnet/data", user), "dev", batch_size, user, total_train)

    hist = model.fit(train_generator,
                     validation_data=val_generator,
                     epochs=num_epochs,
                     # steps_per_epoch=100,
                     callbacks=[early_stopping, reduce_lr], shuffle=True)
    end_time = time.time()
    model.save("models/resnet/rnn_model_2")
    model.save_weights("models/merged/rnn_model_2")
    print("Saved model!")
    plot(hist, num_epochs)

    train_time = end_time - start_time
    total_time += train_time
    print("Total training time: {}".format(total_time))

    with open("models/merged/rnn_model_2/history", "wb") as file:
        pickle.dump(hist.history, file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin', 'all'],
                    help='choose user or train total')
    args = ap.parse_args()
    user = args.users

    train(user)
