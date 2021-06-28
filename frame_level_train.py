import pandas as pd
import numpy as np
import os
import time
import random
import argparse
import matplotlib.pyplot as plt
import cv2
import pickle
from utils import plot, compute_acc
from classification_models.keras import Classifiers
import tensorflow
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dropout, Dense, BatchNormalization, Input, AveragePooling2D, LSTM, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from data_gen import Generator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# physical_devices = tensorflow.config.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

idx2letter = {-1: 'NULL', 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
              11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
              22: 'w', 23: 'x', 24: 'y',25: 'z', 26: 'SIL_START', 27: 'SIL_END'}

num_classes = len(idx2letter)


def get_data(path, image_range):
    df = pd.read_csv("csv/fsvid.csv")
    filename_list = sorted(df['filename'].tolist())
    labels, list_of_images, list_of_labels = [], [], []
    count = 0
    for folder in sorted(os.listdir(path)):
        if folder in filename_list:
            count += 1
            idx = filename_list.index(folder)
            label = df.iloc[idx]['sequence_numbers']
            label = label.strip('][').split(', ')
            label = [int(i) for i in label]
            labels.append(label)
        else:
            continue

        list_of_labels = [item for sublist in labels for item in sublist]
        folder_path = os.path.join(path, folder)

        for images in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, images)
            # vec = np.load(img_path)  # load feature vector for resnet and Hog
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (64, 64))
            img = img / 255.
            list_of_images.append(img)

    images, labels = [], []
    for i in range(0, image_range):
        images.append(list_of_images[i])
        labels.append(list_of_labels[i])

    print(len(list_of_images))
    print(len(list_of_labels))
    return np.array(images), np.array(labels)


def build_model():
    print("Building model")

    base = VGG19(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
    X = base.output
    X = Flatten()(X)
    X = Dense(200, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Dense(100, activation='relu')(X)
    X = Dropout(0.5)(X)
    preds = Dense(num_classes, activation='softmax')(X)
    model = Model(inputs=base.input, outputs=preds)

    sgd = SGD(lr=1e-2, decay=1e-3)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    num_epochs = 10
    batch_size = 16
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin'], required=True,
                    help='choose user or train total')
    args = ap.parse_args()
    user = args.users
    path = os.path.join("data", user)
    mode = input("Train[1] or Test[2]?")

    if mode == '1':
        total_time = 0
        start_time = time.time()
        img_range = 8000

        X_train, y_train = get_data(path + "/train", img_range)
        # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        # y_train = np.reshape(y_train, (y_train.shape[0], 1))

        print(X_train.shape, y_train.shape)

        X_dev, y_dev = get_data(path + "/dev", img_range)
        # X_dev = np.reshape(X_dev, (X_dev.shape[0], 1,  X_dev.shape[1]))
        # y_dev = np.reshape(y_dev, (y_dev.shape[0], 1))

        model = build_model()
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)]
        hist = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=num_epochs, batch_size=batch_size,
                         callbacks=callbacks)
        end_time = time.time()

        model.save("models/frame_level/model_vgg_1/")
        print("Saved model!")
        plot(hist, num_epochs)

        with open("models/frame_level/model_vgg_1/history", "wb") as file:
            pickle.dump(hist.history, file)
        train_time = end_time - start_time
        total_time += train_time
        print("Total training time: {}".format(total_time))

    elif mode == '2':
        img_range = 2000  # < 7227
        X_test, y_test = get_data(path + "/test", img_range)
        # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        print(X_test.shape, y_test.shape)
        model = load_model("models/frame_level/model_vgg_1/")

        preds = model.predict(X_test)
        correct = 0
        count = 0
        for i in range(len(preds)):
            count += 1
            truth = y_test[i]
            pred = int(np.argmax(preds[i], axis=-1).astype(int))

            if truth == pred:
                correct += 1
        print(correct, count, correct/count)

        plt.figure(figsize=(25, 20))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(20):
            truth = idx2letter[y_test[i + 685].astype(int)]
            pred = idx2letter[np.argmax(preds[i + 685], axis=-1).astype(int)]
            print("Truth {} Pred {}".format(truth, pred))
            plt.subplot(4, 5, i + 1)
            plt.imshow(X_test[i+685])
            plt.title('true: {} - pred: {}'.format(truth, pred))
        plt.show()
    else:
        print("Please enter the correct mode. Exiting..")
        exit(-9)
